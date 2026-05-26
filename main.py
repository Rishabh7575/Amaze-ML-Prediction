from src.pipeline import run_pipeline

if __name__ == '__main__':
    # Run pipeline with text features active by default, matching original run settings
    run_pipeline(use_text=True, use_image=False)
