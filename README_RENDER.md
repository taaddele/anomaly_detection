# Deploy the CI/CD CNN–Transformer log classifier on Render

This is the CPU-only Render version of the Gradio app adapted from the supplied training notebook. It expects the same trained checkpoint and character vocabulary.

## Required model artifacts

Place these files in the same repository directory as `app.py`:

- `gha_log_transformer_model.pt`
- `char_vocab.json`

Generate them by running the notebook through its model-saving cell (the checkpoint includes the architecture settings and class names; the app uses the notebook-compatible defaults for `max_chars=256`, a 30-line window, and a 15-line stride).

Do not upload your training dataset. Do not upload real workflow logs containing secrets.

## Option A — deploy using Render Blueprint

1. Create a GitHub repository and upload `app.py`, `requirements.txt`, `render.yaml`, `gha_log_transformer_model.pt`, and `char_vocab.json` to the repository root.
2. In Render, choose **New → Blueprint** and connect that GitHub repository.
3. Render reads `render.yaml`. Review the service configuration and choose the Free plan.
4. Deploy. Open the service Logs and wait for the first build to complete.
5. Open the `onrender.com` URL and test with a short, non-sensitive log file.

## Option B — create a Web Service manually

1. Push the same five files to a GitHub repository.
2. In Render, choose **New → Web Service**, connect the repository, and select the branch.
3. Runtime: **Python**.
4. Build command:

   `pip install --no-cache-dir torch==2.8.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt`

5. Start command:

   `python app.py`

6. Select **Free** compute.
7. Add these environment variables:
   - `PYTHON_VERSION` = `3.11.11`
   - `OMP_NUM_THREADS` = `1`
   - `MKL_NUM_THREADS` = `1`
   - `OPENBLAS_NUM_THREADS` = `1`
   - `TORCH_NUM_THREADS` = `1`
8. Click **Create Web Service**. Render sets `PORT` automatically; the app binds to `0.0.0.0` and uses that port.

## What this app does

- Loads the trained PyTorch checkpoint and saved character vocabulary.
- Uses CPU inference (Render Free does not provide a GPU).
- Retains the notebook's Character-CNN + Transformer model structure and overlapping 30-line windows with a 15-line step.
- Allows either pasted log text or a `.txt` upload.
- Restricts each request to 500,000 characters and 4,000 retained lines to reduce memory pressure on the Free instance.
- Limits model inference to one window batch at a time and one queued request at a time.

## Free-plan caveats

Render Free web services have 512 MB RAM and less than one CPU, and spin down after 15 minutes without inbound traffic. First request after an idle period can take about a minute to wake. Free services have an ephemeral filesystem, but model artifacts committed with the app are part of each deployment and are present at startup. This plan is useful for a research/demo prototype, not production.

The classifier's parameter count is around 0.7 million (roughly 2.7 MB for float32 weights, depending on vocabulary size). The runtime overhead of PyTorch, NumPy, and Gradio is much larger than the model weights; if the service is killed with an out-of-memory error (often exit code 137), Render Free is too small. You would need a larger paid instance, or a more lightweight inference runtime. Render's current pricing lists a 2 GB / 1 CPU instance at $25/month; check the pricing page before upgrading.

## Troubleshooting

- **Missing checkpoint or vocabulary:** verify the two artifact filenames and that they are at repository root.
- **State-dict size mismatch:** make sure the checkpoint came from this exact CNN–Transformer notebook, not the CNN-only or BiLSTM notebook.
- **Out of memory / exit 137:** reduce `MAX_INPUT_CHARACTERS` and `MAX_RETAINED_LINES` further, and keep concurrency at one. If startup itself is killed, the Free plan cannot support the installed runtime; use a larger RAM plan or a smaller deployment stack.
- **Build failure installing torch:** confirm the build command includes the PyTorch CPU wheel index and Python 3.11.
- **Slow prediction:** this service is CPU-only and has less than one CPU on Free. Test with shorter logs.

## Official Render docs

- https://render.com/docs/free
- https://render.com/docs/your-first-deploy
- https://render.com/docs/environment-variables
- https://render.com/pricing
