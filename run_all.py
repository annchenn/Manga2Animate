import magi
import create_character_description
import create_script_withoutfeedback
import evaluate_scripts
import create_script_withfeedback
import generate_storyboard
import evaluate_storyboard


def main():
    # --- Step 0: Extract transcript and character description ---
    print("=" * 60)
    print("STEP 0: Extracting transcript with MAGI")
    print("=" * 60)
    magi.main()

    print("=" * 60)
    print("STEP 0: Generating character description")
    print("=" * 60)
    create_character_description.main()

    # --- Iteration 0: generate scripts without feedback ---
    print("=" * 60)
    print("ITERATION 0: Generating initial scripts (no feedback)")
    print("=" * 60)
    create_script_withoutfeedback.main(iteration=0)

    print("=" * 60)
    print("ITERATION 0: Evaluating scripts")
    print("=" * 60)
    evaluate_scripts.main(iteration=0)

    # --- Iteration 1: refine with feedback from iteration 0 ---
    print("=" * 60)
    print("ITERATION 1: Generating scripts (with feedback)")
    print("=" * 60)
    create_script_withfeedback.main(idx=0, iteration=1)

    print("=" * 60)
    print("ITERATION 1: Evaluating scripts")
    print("=" * 60)
    evaluate_scripts.main(iteration=1)

    # --- Iteration 2: refine with feedback from iteration 1 ---
    print("=" * 60)
    print("ITERATION 2: Generating scripts (with feedback)")
    print("=" * 60)
    create_script_withfeedback.main(idx=1, iteration=2)

    print("=" * 60)
    print("ITERATION 2: Evaluating scripts")
    print("=" * 60)
    evaluate_scripts.main(iteration=2)

    # --- Generate storyboard from final best script ---
    print("=" * 60)
    print("Generating storyboards")
    print("=" * 60)
    for idx in range(3):
        print(f"\n=== Storyboard generation for bestScript{idx}.json ===")
        generate_storyboard.main(idx=idx)

    print("=" * 60)
    print("Evaluating storyboards")
    print("=" * 60)
    evaluate_storyboard.main()

    print("=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
