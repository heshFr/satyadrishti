import { motion } from "framer-motion";

interface ShieldBreathingProps {
  status: "idle" | "safe" | "danger";
}

const ShieldBreathing = ({ status }: ShieldBreathingProps) => {
  if (status === "danger") return null;

  return (
    <div className="relative flex items-center justify-center">
      {/* Outer pulse rings */}
      <motion.div
        className={`absolute w-40 h-40 rounded-full ${
          status === "safe" ? "bg-neon-green/10" : "bg-primary/10"
        }`}
        animate={{ scale: [1, 1.4], opacity: [0.3, 0] }}
        transition={{ duration: 3, repeat: Infinity, ease: "easeOut" }}
      />
      <motion.div
        className={`absolute w-32 h-32 rounded-full ${
          status === "safe" ? "bg-neon-green/15" : "bg-primary/15"
        }`}
        animate={{ scale: [1, 1.3], opacity: [0.4, 0] }}
        transition={{
          duration: 3,
          repeat: Infinity,
          ease: "easeOut",
          delay: 0.5,
        }}
      />

      {/* Logo image */}
      <motion.div
        animate={{ scale: [1, 1.06, 1] }}
        transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
      >
        <div className="relative">
          <img
            src="/logo.png"
            alt="Satya Drishti"
            className="w-24 h-24 object-contain drop-shadow-[0_0_15px_rgba(0,209,255,0.4)]"
          />
          <motion.div
            className={`absolute inset-0 ${
              status === "safe" ? "bg-neon-green/20" : "bg-primary/10"
            } blur-2xl rounded-full`}
            animate={{ opacity: [0.2, 0.5, 0.2] }}
            transition={{ duration: 3, repeat: Infinity }}
          />
        </div>
      </motion.div>
    </div>
  );
};

export default ShieldBreathing;
