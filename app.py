    n_lines = len(items)
    if n_lines <= WINDOW_SIZE:
        starts = [0]
    else:
        starts = list(range(0, n_lines - WINDOW_SIZE + 1, WINDOW_STEP))
        last_start = n_lines - WINDOW_SIZE
        if not starts or starts[-1] != last_start:
            starts.append(last_start)

    # Average logits for a line over every overlapping window containing it,
    # matching the notebook's predict_unique_lines evaluation procedure.
    score_sums = [np.zeros(NUM_CLASSES, dtype=np.float64) for _ in range(n_lines)]
    score_counts = np.zeros(n_lines, dtype=np.int64)
    model.eval()

    with torch.inference_mode():
        for batch_start in range(0, len(starts), INFERENCE_BATCH_WINDOWS):
            batch_starts = starts[batch_start:batch_start + INFERENCE_BATCH_WINDOWS]
            batch_x = np.zeros(
                (len(batch_starts), WINDOW_SIZE, MAX_CHARS), dtype=np.int64
            )
            batch_padding = np.ones(
                (len(batch_starts), WINDOW_SIZE), dtype=bool
            )
            batch_line_indices = np.full(
                (len(batch_starts), WINDOW_SIZE), -1, dtype=np.int64
            )

            for batch_i, window_start in enumerate(batch_starts):
                for position in range(WINDOW_SIZE):
                    line_i = window_start + position
                    if line_i >= n_lines:
                        break
                    char_ids = [
                        char_to_id.get(ch, UNK_ID)
                        for ch in items[line_i]["clean"][:MAX_CHARS]
                    ]
                    if char_ids:
                        batch_x[batch_i, position, :len(char_ids)] = char_ids
                    batch_padding[batch_i, position] = False
                    batch_line_indices[batch_i, position] = line_i

            x_tensor = torch.from_numpy(batch_x).to(DEVICE)
            padding_tensor = torch.from_numpy(batch_padding).to(DEVICE)
            batch_logits = model(x_tensor, padding_tensor).detach().cpu().numpy()

            for batch_i in range(len(batch_starts)):
                for position in range(WINDOW_SIZE):
                    line_i = int(batch_line_indices[batch_i, position])
                    if line_i < 0:
                        continue
                    score_sums[line_i] += batch_logits[batch_i, position]
                    score_counts[line_i] += 1

    rows = []
    predicted_counts = Counter()
    anomaly_lines = 0
    top_scores = []

    for i, item in enumerate(items):
        mean_logits = score_sums[i] / max(int(score_counts[i]), 1)
        probabilities = _softmax(mean_logits)
        pred_id = int(np.argmax(mean_logits))
        confidence_like_score = float(probabilities[pred_id])
        predicted_class = CLASS_NAMES[pred_id]
        predicted_counts[predicted_class] += 1
        anomaly = pred_id != 0
        anomaly_lines += int(anomaly)
        top_scores.append(confidence_like_score)
        rows.append([
            item["line_number"],
            predicted_class,
            round(confidence_like_score * 100, 2),
            "ANOMALY" if anomaly else "NORMAL",
            item["clean"][:300],
        ])

    count_text = "\n".join(
        f"- **{name}:** {predicted_counts.get(name, 0)} line(s)"
        for name in CLASS_NAMES
    )
    summary = (
        "### Analysis summary\n"
        f"- Non-empty log lines classified: **{n_lines:,}**\n"
        f"- Lines predicted anomalous: **{anomaly_lines:,}**\n"
        f"- Lines predicted normal: **{n_lines - anomaly_lines:,}**\n"
        f"- Mean top-class score: **{100 * float(np.mean(top_scores)):.2f}%**\n\n"
        "### Predicted class counts\n"
        f"{count_text}\n\n"
        "> Scores are softmax outputs, not calibrated probabilities. Predictions are model outputs, "
        "not proof that an incident exists. Each line is classified using its surrounding "
        "log-window context."
    )
    return rows, summary


def analyze_uploaded_or_pasted_log(pasted_text, uploaded_file):
    """Gradio callback. An uploaded .txt file takes precedence over pasted text."""
    try:
        if uploaded_file is not None:
            file_path = uploaded_file if isinstance(uploaded_file, str) else uploaded_file.name
            raw_text = Path(file_path).read_text(encoding="utf-8", errors="replace")
        else:
            raw_text = pasted_text or ""
        return predict_log_text(raw_text)
    except Exception as exc:
        return [], f"**Unable to analyze this input:** {type(exc).__name__}: {exc}"


with gr.Blocks(title="CI/CD Workflow Log Anomaly Classifier") as demo:
    gr.Markdown(
        "# CI/CD Workflow Log Anomaly Classifier\n"
        "Paste a GitHub Actions/CI log or upload a `.txt` file. The model classifies "
        "retained log lines as Normal, Flaky test, Configuration drift, Security anomaly, "
        "or Silent failure. Predictions use neighbouring log lines as context."
    )
    with gr.Row():
        pasted_input = gr.Textbox(
            label="Paste workflow log text",
            placeholder="Run npm test\nError: expected status 200, received 500\nBuild failed",
            lines=16,
            scale=2,
        )
        file_input = gr.File(
            label="Or upload a .txt log file (takes precedence)",
            file_types=[".txt"],
            type="filepath",
            scale=1,
        )
    analyze_button = gr.Button("Analyze log", variant="primary")
    output_table = gr.Dataframe(
        headers=["Input line #", "Predicted class", "Top score (%)", "Decision", "Cleaned log text"],
        datatype=["number", "str", "number", "str", "str"],
        label="Line-by-line predictions",
        interactive=False,
        wrap=True,
    )
    summary_output = gr.Markdown(label="Summary")
    gr.Markdown(
        "### Important notes\n"
        "- The displayed score is not a calibrated probability.\n"
        "- A predicted anomaly is a review signal, not proof of a security incident or failure.\n"
        "- Do not paste access tokens, passwords, private keys, or other sensitive log content into a public demo."
    )
    analyze_button.click(
        fn=analyze_uploaded_or_pasted_log,
        inputs=[pasted_input, file_input],
        outputs=[output_table, summary_output],
    )

if __name__ == "__main__":
    # Render requires the service to bind to 0.0.0.0 and the PORT it provides.
    port = int(os.environ.get("PORT", "10000"))
    demo.queue(max_size=4, default_concurrency_limit=1)
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=False,
        max_threads=2,
