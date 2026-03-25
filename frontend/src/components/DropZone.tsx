import { useState, useRef, useCallback, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { motion, AnimatePresence } from "framer-motion";
import MaterialIcon from "./MaterialIcon";

interface DropZoneProps {
  onFileSelect: (file: File) => void;
  selectedFile: File | null;
  onClear?: () => void;
}

const ACCEPTED_TYPES = [
  "image/jpeg",
  "image/png",
  "image/webp",
  "video/mp4",
  "video/quicktime",
  "video/x-msvideo",
];

const DropZone = ({ onFileSelect, selectedFile, onClear }: DropZoneProps) => {
  const { t } = useTranslation();
  const [isDragging, setIsDragging] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const handleFile = useCallback(
    (file: File) => {
      if (!ACCEPTED_TYPES.includes(file.type)) return;
      if (file.size > 100 * 1024 * 1024) return;
      onFileSelect(file);
      if (file.type.startsWith("image/")) {
        const url = URL.createObjectURL(file);
        setPreview(url);
      } else {
        setPreview(null);
      }
    },
    [onFileSelect]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
    },
    [handleFile]
  );

  const formatSize = (bytes: number) => {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  // File selected state
  if (selectedFile) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 10 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        className="relative rounded-xl bg-surface-container-low p-6 flex items-center gap-5 border border-outline-variant/10"
      >
        <div className="w-20 h-20 rounded-xl bg-surface-container-high flex items-center justify-center overflow-hidden shrink-0">
          {preview ? (
            <img src={preview} alt="Preview" className="w-full h-full object-cover" />
          ) : (
            <MaterialIcon icon="video_file" size={32} className="text-primary/50" />
          )}
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-on-surface font-headline font-semibold truncate">{selectedFile.name}</p>
          <p className="text-sm text-on-surface-variant mt-1 font-label">
            {formatSize(selectedFile.size)} · {selectedFile.type.startsWith("image/") ? "Image" : "Video"}
          </p>
        </div>
        {onClear && (
          <button
            onClick={() => {
              onClear();
              setPreview(null);
            }}
            className="p-2 rounded-lg text-on-surface-variant hover:text-error hover:bg-error-container/20 transition-colors"
          >
            <MaterialIcon icon="close" size={20} />
          </button>
        )}
      </motion.div>
    );
  }

  // Drop zone state
  return (
    <motion.div
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
      className="relative cursor-pointer group"
    >
      <motion.div
        animate={{
          borderColor: isDragging ? "rgba(0, 209, 255, 0.4)" : "rgba(255, 255, 255, 0.06)",
        }}
        transition={{ duration: 0.3 }}
        className="rounded-xl border-2 border-dashed p-14 flex flex-col items-center justify-center gap-5 transition-all duration-300 group-hover:border-primary/30 group-hover:bg-primary/[0.02] relative overflow-hidden bg-surface-container-low"
      >
        {/* Background glow when dragging */}
        <AnimatePresence>
          {isDragging && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 bg-gradient-to-br from-primary/[0.06] to-secondary/[0.04]"
            />
          )}
        </AnimatePresence>

        {/* Upload icon */}
        <motion.div
          animate={{ y: isDragging ? -8 : 0, scale: isDragging ? 1.1 : 1 }}
          transition={{ type: "spring", stiffness: 300 }}
          className="relative"
        >
          <div className="w-20 h-20 rounded-2xl bg-surface-container-high flex items-center justify-center group-hover:shadow-glow-emerald transition-shadow duration-500">
            {isDragging ? (
              <MaterialIcon icon="cloud_upload" size={40} className="text-primary" />
            ) : (
              <MaterialIcon icon="upload_file" size={40} className="text-primary/70 group-hover:text-primary transition-colors" />
            )}
          </div>
          {/* Floating dot */}
          <motion.div
            animate={{ y: [-3, 3, -3], opacity: [0.4, 0.8, 0.4] }}
            transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
            className="absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full bg-primary shadow-[0_0_8px_rgba(0,209,255,0.6)]"
          />
        </motion.div>

        <div className="text-center relative z-10">
          <p className="text-on-surface font-headline font-semibold text-lg">
            {isDragging ? "Drop your file here" : t("scanner.dropzoneTitle")}
          </p>
          <p className="text-on-surface-variant text-sm mt-1.5 font-body">
            {t("scanner.dropzoneSubtitle")}
          </p>
        </div>

        {/* Format badges */}
        <div className="flex flex-wrap items-center justify-center gap-2 mt-1 relative z-10">
          {[
            { icon: "image", label: "JPG, PNG, WEBP" },
            { icon: "videocam", label: "MP4, MOV, AVI" },
          ].map((fmt) => (
            <span
              key={fmt.label}
              className="flex items-center gap-1.5 text-xs text-on-surface-variant/70 px-3 py-1.5 rounded-full bg-surface-container-high border border-outline-variant/10 font-label"
            >
              <MaterialIcon icon={fmt.icon} size={14} /> {fmt.label}
            </span>
          ))}
          <span className="text-xs text-on-surface-variant/50 px-3 py-1.5 rounded-full bg-surface-container-high border border-outline-variant/10 font-label">
            Max 100MB
          </span>
        </div>
      </motion.div>

      <input
        ref={inputRef}
        type="file"
        accept=".jpg,.jpeg,.png,.webp,.mp4,.mov,.avi"
        className="hidden"
        onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
      />
    </motion.div>
  );
};

export default DropZone;
