export default function Toast({ toast }) {
  if (!toast) {
    return null;
  }

  return (
    <div className={`toast toast-${toast.tone}`}>
      {toast.message}
    </div>
  );
}
