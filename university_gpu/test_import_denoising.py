import importlib
import sys
import os
import traceback


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def try_import(module_name):
    print(f"\nTrying to import: {module_name}")

    try:
        module = importlib.import_module(module_name)
        print(f"[OK] Successfully imported {module_name}")
        print(f"Module file: {getattr(module, '__file__', '<no file>')}")
        return True
    except Exception as e:
        print(f"[FAILED] Could not import {module_name}")
        print("Error:", repr(e))
        traceback.print_exc()
        return False


def main():
    print_section("Basic Information")
    print("Python executable:", sys.executable)
    print("Python version:", sys.version)
    print("Current working directory:", os.getcwd())

    print_section("Python Path")
    for path in sys.path:
        print(path)

    print_section("Import TTT-Discover Core Package")

    core_modules = [
        "ttt_discover",
        "ttt_discover.discovery",
        "ttt_discover.environments.base_reward_evaluator",
        "ttt_discover.environments.sandbox_reward_evaluator",
    ]

    core_results = []
    for module_name in core_modules:
        core_results.append(try_import(module_name))

    print_section("Import Section 4.4 Denoising Modules")

    denoising_modules = [
        "examples.denoising",
        "examples.denoising.env",
        "examples.denoising.prompt",
        "examples.denoising.utils",
    ]

    denoising_results = []
    for module_name in denoising_modules:
        denoising_results.append(try_import(module_name))

    print_section("Import Results Denoising Modules")

    possible_result_modules = [
        "results.denoising",
        "results.denoising.denoise_ttt",
    ]

    result_import_results = []
    for module_name in possible_result_modules:
        result_import_results.append(try_import(module_name))

    print_section("Summary")

    total_checks = len(core_results) + len(denoising_results) + len(result_import_results)
    passed_checks = sum(core_results) + sum(denoising_results) + sum(result_import_results)

    print(f"Passed {passed_checks}/{total_checks} import checks.")

    if passed_checks == total_checks:
        print("[SUCCESS] All import checks passed.")
    else:
        print("[WARNING] Some imports failed.")
        print("This may be due to missing dependencies.")
        print("If denoising imports fail, we may need to install denoising-specific requirements.")


if __name__ == "__main__":
    main()