interface MaterialIconProps {
  icon: string;
  filled?: boolean;
  className?: string;
  size?: number;
  weight?: number;
}

const MaterialIcon = ({ icon, filled = false, className = "", size = 24, weight = 300 }: MaterialIconProps) => {
  return (
    <span
      className={`material-symbols-outlined ${className}`}
      style={{
        fontVariationSettings: `'FILL' ${filled ? 1 : 0}, 'wght' ${weight}, 'GRAD' 0, 'opsz' ${size}`,
        fontSize: size,
      }}
    >
      {icon}
    </span>
  );
};

export default MaterialIcon;
