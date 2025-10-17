# --- Safety block: limits, pre-checks, watchdog, OOM handling, checkpoints ---
import os, sys, threading, time, psutil, signal, logging
import torch

# --- Logging simple ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("safety")

# --- Limit CPU threads (OpenMP/MKL/PyTorch) ---
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
torch.set_num_threads(4)

# --- Parameters (à adapter selon la machine) ---
MAX_RAM_GB = 12         # si la RAM utilisée par le processus dépasse -> shutdown
MAX_VRAM_GB = 12        # pour GPU, si disponible
CHECK_INTERVAL_SEC = 5
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def setup(max_ram=12, max_vram=12, threads=4):
    global MAX_RAM_GB, MAX_VRAM_GB
    MAX_RAM_GB = max_ram
    MAX_VRAM_GB = max_vram
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    torch.set_num_threads(threads)
    logger.info(f"Safety setup: RAM={max_ram}GB, VRAM={max_vram}GB, threads={threads}")

# --- Pre-check available system memory BEFORE big allocations ---
avail_gb = psutil.virtual_memory().available / 1e9
if avail_gb < 4:  # si trop faible, on stoppe (tweak)
    logger.error(f"RAM disponible trop faible: {avail_gb:.2f} GB. Abandon.")
    sys.exit(1)

# --- Helper to save a checkpoint quickly ---
def save_checkpoint(state, name="checkpoint_safe.pt"):
    try:
        path = os.path.join(CHECKPOINT_DIR, name)
        torch.save(state, path)
        logger.info(f"Checkpoint saved to {path}")
    except Exception as e:
        logger.exception("Erreur en sauvegardant checkpoint: %s", e)

# --- Watchdog thread: surveille RAM et VRAM et exit si dépassement ---
def safety_watchdog(max_ram_gb=MAX_RAM_GB, max_vram_gb=MAX_VRAM_GB, interval=CHECK_INTERVAL_SEC):
    proc = psutil.Process(os.getpid())
    while True:
        try:
            mem_rss_gb = proc.memory_info().rss / 1e9
            if mem_rss_gb > max_ram_gb:
                logger.error(f"Watchdog: RAM used {mem_rss_gb:.2f} GB > {max_ram_gb} GB. Saving checkpoint and exiting.")
                # tentative de checkpoint léger : save un tensor vide ou état minimal
                save_checkpoint({"note": "killed_by_watchdog", "time": time.time()}, name=f"watchdog_preexit_{int(time.time())}.pt")
                os._exit(1)  # exit brutal pour être sûr que tout s'arrête
            # VRAM check si CUDA dispo
            if torch.cuda.is_available():
                try:
                    devid = torch.cuda.current_device()
                    vram_alloc_gb = torch.cuda.memory_allocated(devid) / 1e9
                    if vram_alloc_gb > max_vram_gb:
                        logger.error(f"Watchdog: VRAM used {vram_alloc_gb:.2f} GB > {max_vram_gb} GB. Emptying cache and exiting.")
                        save_checkpoint({"note": "killed_by_watchdog_vram", "time": time.time()}, name=f"watchdog_vram_preexit_{int(time.time())}.pt")
                        torch.cuda.empty_cache()
                        os._exit(1)
                except Exception:
                    # en cas de problème interrogons nvidia-smi (optionnel)
                    pass
        except Exception:
            logger.exception("Erreur dans watchdog")
        time.sleep(interval)

threading.Thread(target=safety_watchdog, daemon=True).start()

# --- Helper pour catcher OOMs lors d'opérations PyTorch ---
def attempt(fn, *args, **kwargs):
    """
    Exécute fn(*args, **kwargs). En cas d'OOM PyTorch, essaye de vider le cache, sauvegarder checkpoint, et ré-élever.
    """
    try:
        return fn(*args, **kwargs)
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or "cuda out of memory" in msg:
            logger.error("Caught OOM during operation. Saving checkpoint and freeing cache.")
            save_checkpoint({"note": "oom_caught", "time": time.time()}, name=f"oom_{int(time.time())}.pt")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # ré-élever l'exception pour gestion ultérieure ou arrêter
            raise
        else:
            raise