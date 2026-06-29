import { useRef } from "react";

function ActionButton({
  icon,
  label,
  onClick,
  disabled = false,
  loading = false,
  variant = "secondary"
}) {
  const className = ["action-button", `action-button-${variant}`];

  if (disabled) {
    className.push("is-disabled");
  }

  return (
    <button
      type="button"
      className={className.join(" ")}
      onClick={onClick}
      disabled={disabled}
    >
      {icon ? <img src={icon} alt="" aria-hidden="true" /> : null}
      <span>{loading ? "Predicting..." : label}</span>
    </button>
  );
}

export default function ActionBar({
  models,
  selectedModelId,
  onModelChange,
  onPredict,
  onUpload,
  predictDisabled,
  loading
}) {
  const inputRef = useRef(null);

  const handleBrowse = () => {
    inputRef.current?.click();
  };

  const handleFilesSelected = (event) => {
    const files = Array.from(event.target.files || []);
    if (files.length > 0) {
      onUpload(files);
    }
    event.target.value = "";
  };

  return (
    <section className="action-bar card">
      <div className="action-group">
        <ActionButton
          icon="/elements/predict_symbol.png"
          label="Predict"
          onClick={onPredict}
          disabled={predictDisabled}
          loading={loading}
          variant="primary"
        />
        <ActionButton
          icon="/elements/upload_symbol.png"
          label="Upload"
          onClick={handleBrowse}
        />
        <input
          ref={inputRef}
          type="file"
          accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm,video/x-m4v,.mp4,.mov,.avi,.mkv,.webm,.m4v"
          multiple
          hidden
          onChange={handleFilesSelected}
        />
      </div>

      <label className="model-select">
        <span className="model-select-icon">
          <img src="/elements/models_symbol.png" alt="" aria-hidden="true" />
        </span>
        <select
          value={selectedModelId}
          onChange={(event) => onModelChange(event.target.value)}
        >
          {models.map((model) => (
            <option key={model.id} value={model.id}>
              {model.label}
            </option>
          ))}
        </select>
      </label>
    </section>
  );
}
