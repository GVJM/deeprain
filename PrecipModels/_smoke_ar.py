"""Smoke test: 5 epochs for each AR model."""
import subprocess, sys

models = [
    "ar_vae", "ar_flow_match", "ar_latent_fm",
    "ar_real_nvp", "ar_real_nvp_lstm",
    "ar_glow", "ar_glow_lstm",
    "ar_mean_flow", "ar_mean_flow_lstm",
    "ar_flow_map", "ar_flow_map_lstm",
]

passed, failed = [], []
for m in models:
    print(f"\n{'='*40}\n=== {m} ===\n{'='*40}")
    result = subprocess.run(
        [sys.executable, "train.py",
         "--model", m,
         "--data_path", "../dados_sabesp/dayprecip.dat",
         "--max_epochs", "5",
         "--holdout_ratio", "0"],
        capture_output=False,
    )
    if result.returncode == 0:
        passed.append(m)
        print(f">>> {m} OK")
    else:
        failed.append(m)
        print(f">>> {m} FAILED (exit code {result.returncode})")

print("\n" + "="*50)
print(f"PASSED ({len(passed)}): {passed}")
print(f"FAILED ({len(failed)}): {failed}")
