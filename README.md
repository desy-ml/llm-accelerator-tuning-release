# LLM Accelerator Tuning

To run accelerator tuning utilising our LLM-based approach and generate ypur own data for the latter, execute any of the `try_langchain_*.ipynb` notebooks, each of which is designed to use a different one of the presented prompts.
Note that you will have to have an OpenAI API key or a running instance of Ollama, and that you will need to point to them in the respective locations.

To generate the data for the baseline algorithms, run `publications/paper/generate_baseline_data.ipynb`.

To compute the results from our experiments as presented in the paper, run `publications/paper/results_table.ipynb` with our data placed in the `data/paper/` directory.
