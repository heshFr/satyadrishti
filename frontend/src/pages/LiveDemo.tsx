import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import MaterialIcon from "@/components/MaterialIcon";

const LiveDemo = () => {
  return (
    <div className="min-h-screen bg-background relative overflow-hidden flex flex-col items-center justify-center p-6">
      {/* Cinematic Background */}
      <div className="absolute inset-0 z-0">
        <div className="absolute inset-0 bg-gradient-to-b from-black/80 via-black/40 to-black/90 z-10" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-primary/10 via-transparent to-transparent opacity-50 blur-3xl" />
      </div>

      {/* Navigation Header */}
      <header className="absolute top-0 w-full p-8 flex justify-between items-center z-50">
        <Link to="/" className="flex items-center gap-3 group">
          <div className="w-10 h-10 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center group-hover:bg-primary/20 group-hover:border-primary/40 transition-all active:scale-95">
            <MaterialIcon icon="arrow_back" size={24} className="text-on-surface-variant group-hover:text-primary transition-colors" />
          </div>
          <span className="font-headline text-sm font-black uppercase tracking-widest text-on-surface-variant group-hover:text-on-surface transition-colors">Return to Base</span>
        </Link>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 px-4 py-2 bg-error/10 border border-error/20 rounded-full">
            <div className="w-2 h-2 rounded-full bg-error animate-pulse" />
            <span className="text-[10px] font-black text-error uppercase tracking-widest">Live Forensic Demonstration</span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <motion.div 
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
        className="relative z-10 w-full max-w-6xl aspect-video rounded-[2rem] overflow-hidden border border-white/10 shadow-[0_0_80px_rgba(0,0,0,0.5)] bg-surface-container-low group"
      >
        <video 
          src="/video.mp4" 
          controls 
          autoPlay 
          className="w-full h-full object-cover"
          poster="/LandingBackground.png"
        >
          Your browser does not support the video tag.
        </video>

        {/* Forensic Overlay (Visual Only) */}
        <div className="absolute inset-0 pointer-events-none border-[20px] border-black/20 z-20" />
        <div className="absolute top-10 left-10 z-20 opacity-40">
           <div className="flex flex-col gap-1">
             <div className="h-px w-20 bg-primary" />
             <span className="font-mono text-[8px] text-primary uppercase tracking-[0.4em]">Intercept.v9.4</span>
           </div>
        </div>
        <div className="absolute bottom-10 right-10 z-20 opacity-40">
           <div className="flex flex-col items-end gap-1">
             <span className="font-mono text-[8px] text-primary uppercase tracking-[0.4em]">Deep-Neural-Audit</span>
             <div className="h-px w-20 bg-primary" />
           </div>
        </div>
      </motion.div>

      {/* Footer Copy */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5, duration: 0.8 }}
        className="mt-12 text-center space-y-4 z-10"
      >
        <h2 className="font-headline text-3xl font-black text-on-surface uppercase tracking-tight">Witness the Defense Intelligence</h2>
        <p className="text-on-surface-variant max-w-xl text-lg font-light italic">
          "This demonstration showcases the real-time intercept and forensic dissection of a synthetic media injection attempt."
        </p>
      </motion.div>
    </div>
  );
};

export default LiveDemo;
