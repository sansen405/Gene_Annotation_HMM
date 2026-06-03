import { useEffect, useRef } from "react";
import { createPortal } from "react-dom";

export function ExplorerContextMenu({ menu, onClose, onDelete, onDownload, onRename }) {
  const menuRef = useRef(null);
  const showDownload = menu?.entry?.type === "file" && onDownload;

  useEffect(() => {
    if (!menu) return undefined;

    function handlePointerDown(event) {
      if (menuRef.current?.contains(event.target)) return;
      onClose();
    }

    function handleKeyDown(event) {
      if (event.key === "Escape") onClose();
    }

    window.addEventListener("pointerdown", handlePointerDown);
    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("scroll", onClose, true);
    window.addEventListener("resize", onClose);

    return () => {
      window.removeEventListener("pointerdown", handlePointerDown);
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("scroll", onClose, true);
      window.removeEventListener("resize", onClose);
    };
  }, [menu, onClose]);

  if (!menu) return null;

  return createPortal(
    <div
      className="explorer-context-menu"
      ref={menuRef}
      role="menu"
      style={{ left: menu.x, top: menu.y }}
    >
      <button
        className="explorer-context-menu-item"
        onClick={() => {
          onRename(menu.entry);
          onClose();
        }}
        role="menuitem"
        type="button"
      >
        Rename
      </button>
      <button
        className="explorer-context-menu-item explorer-context-menu-item--danger"
        onClick={() => {
          onDelete(menu.entry);
          onClose();
        }}
        role="menuitem"
        type="button"
      >
        Delete
      </button>
      {showDownload ? (
        <button
          className="explorer-context-menu-item"
          onClick={() => {
            onDownload(menu.entry);
            onClose();
          }}
          role="menuitem"
          type="button"
        >
          Download
        </button>
      ) : null}
    </div>,
    document.body
  );
}
