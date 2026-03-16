# TMLR FutureScope - One-Command Workflow

.PHONY: all data benchmark docker clean help

all: data benchmark  ## Complete TMLR pipeline (data + benchmark)

data:  ## Download real datasets
	@echo "Downloading real public datasets..."
	python download_real_data.py
	python fix_electricity.py
	@echo "✓ Datasets ready"

benchmark: data  ## Run TMLR benchmark (70 seconds)
	@echo "Running TMLR final benchmark..."
	python -u benchmark_tmlr_final.py
	@echo "✓ Results saved to benchmark_tmlr_real.csv"

docker:  ## Build Docker container
	@echo "Building TMLR Docker image..."
	docker build -t tmlr-futurescope .
	@echo "✓ Run with: docker run -v $(PWD)/results:/app/results tmlr-futurescope"

docker-run:  ## Run benchmark in Docker
	docker run -v $(PWD)/results:/app/results tmlr-futurescope

clean:  ## Remove generated files
	rm -rf data/real figures_tmlr_real benchmark_tmlr_real.csv
	find . -type d -name __pycache__ -exec rm -rf {} +
	@echo "✓ Cleaned"

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

# TMLR Submission Workflow:
# 1. make data        # Download datasets
# 2. make benchmark   # Run benchmark (70s)
# 3. Check: benchmark_tmlr_real.csv + figures_tmlr_real/
