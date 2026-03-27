import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { useTranslation } from "react-i18next";
import MaterialIcon from "@/components/MaterialIcon";

/* ── Stagger animations ── */
const stagger = {
  container: {
    animate: {
      transition: { staggerChildren: 0.15, delayChildren: 0.2 },
    },
  },
  item: {
    initial: { opacity: 0, y: 40, scale: 0.95 },
    animate: { 
      opacity: 1, 
      y: 0, 
      scale: 1, 
      transition: { duration: 0.7, ease: [0.16, 1, 0.3, 1] as const } 
    },
  },
};

const Hub = () => {
  const { t } = useTranslation();

  return (
    <div className="min-h-screen bg-surface flex flex-col items-center justify-center relative overflow-hidden font-body selection:bg-primary/30 selection:text-primary-container p-6 md:p-12">
      
      {/* ── Cinematic Background ── */}
      <div className="absolute inset-0 pointer-events-none z-0">
        <video 
           autoPlay 
           loop 
           muted 
           playsInline 
           className="w-full h-full object-cover opacity-[0.1]"
        >
          {/* Optional fallback, but the CSS gradient and glow mask works too */}
        </video>
        <div className="absolute inset-0 backdrop-blur-[100px] bg-surface/80" />
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-surface/50 to-surface pointer-events-none" />
        
        {/* Animated Glow Orbs */}
        <motion.div 
          className="absolute top-[-10%] left-[-10%] w-[500px] h-[500px] bg-primary/20 rounded-full blur-[120px]" 
          animate={{ x: [0, 100, 0], y: [0, 50, 0] }}
          transition={{ duration: 15, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.div 
          className="absolute bottom-[-10%] right-[-10%] w-[600px] h-[600px] bg-secondary/10 rounded-full blur-[150px]"
          animate={{ x: [0, -100, 0], y: [0, -50, 0] }}
          transition={{ duration: 20, repeat: Infinity, ease: "easeInOut" }}
        />
      </div>

      <div className="relative z-10 w-full max-w-7xl mx-auto flex flex-col items-center">
        {/* ── Header ── */}
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center space-y-4 mb-16"
        >
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-surface-container/50 border border-outline-variant/10 shadow-[0_0_30px_rgba(0,209,255,0.1)] mb-6">
            <MaterialIcon icon="visibility" size={32} className="text-primary" />
          </div>
          <h1 className="text-5xl md:text-7xl font-headline font-extrabold tracking-tighter text-on-surface">
            Deploy <span className="text-primary-container">Agent Module</span>
          </h1>
          <p className="text-xl md:text-2xl text-on-surface-variant font-light max-w-2xl mx-auto">
            Select the compliance agent you wish to activate. All processing runs locally with full audit logging.
          </p>
        </motion.div>

        {/* ── Option Bubbles ── */}
        <motion.div 
          variants={stagger.container}
          initial="initial"
          animate="animate"
          className="grid grid-cols-1 md:grid-cols-3 gap-8 md:gap-12 w-full max-w-6xl"
        >
          {/* Card 1: Dashboard */}
          <motion.div variants={stagger.item} className="h-full">
            <Link to="/" className="group block relative h-full bg-surface-container-low/40 backdrop-blur-xl border border-outline-variant/20 rounded-[2.5rem] p-10 hover:border-outline/50 transition-all duration-500 hover:-translate-y-2 hover:shadow-[0_30px_60px_rgba(0,0,0,0.2)] overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-br from-on-surface/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
              <div className="absolute top-0 left-0 w-full h-[2px] bg-gradient-to-r from-transparent via-white/20 to-transparent transform scale-x-0 group-hover:scale-x-100 transition-transform duration-700" />
              
              <div className="relative z-10 flex flex-col h-full gap-6">
                <div className="w-20 h-20 rounded-2xl bg-surface/50 border border-outline-variant/30 flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform duration-500 ease-out">
                  <MaterialIcon icon="dashboard" size={36} className="text-on-surface" />
                </div>
                <div className="space-y-4 mt-auto">
                  <h2 className="text-3xl font-headline font-bold text-on-surface">Dashboard</h2>
                  <p className="text-on-surface-variant font-light text-lg">
                    Return to the main overview and see high-level compliance metrics and agent status.
                  </p>
                </div>
              </div>
            </Link>
          </motion.div>

          {/* Card 2: Call Protection */}
          <motion.div variants={stagger.item} className="h-full">
            <Link to="/call-protection" className="group block relative h-full bg-surface-container-low/40 backdrop-blur-xl border border-outline-variant/20 rounded-[2.5rem] p-10 hover:border-primary/40 transition-all duration-500 hover:-translate-y-2 hover:shadow-[0_30px_60px_rgba(0,209,255,0.15)] overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-br from-primary/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
              <div className="absolute top-0 left-0 w-full h-[2px] bg-gradient-to-r from-transparent via-primary/50 to-transparent transform scale-x-0 group-hover:scale-x-100 transition-transform duration-700" />
              
              {/* Highlight Glow bg */}
              <div className="absolute -bottom-20 -right-20 w-64 h-64 bg-primary/20 rounded-full blur-[80px] opacity-0 group-hover:opacity-100 transition-opacity duration-700" />

              <div className="relative z-10 flex flex-col h-full gap-6">
                <div className="w-20 h-20 rounded-2xl bg-primary/20 border border-primary/30 flex items-center justify-center shadow-[0_0_20px_rgba(0,209,255,0.2)] group-hover:scale-110 transition-transform duration-500 ease-out">
                  <MaterialIcon icon="record_voice_over" size={36} className="text-primary" filled />
                </div>
                <div className="space-y-4 mt-auto">
                  <h2 className="text-3xl font-headline font-bold text-on-surface">Voice Forensics Agent</h2>
                  <p className="text-on-surface-variant font-light text-lg">
                    Activate real-time 9-layer voice forensics with Biological Veto to detect AI clones and fraud during live calls.
                  </p>
                </div>
              </div>
            </Link>
          </motion.div>

          {/* Card 3: Media Scanner */}
          <motion.div variants={stagger.item} className="h-full">
            <Link to="/scanner" className="group block relative h-full bg-surface-container-low/40 backdrop-blur-xl border border-outline-variant/20 rounded-[2.5rem] p-10 hover:border-secondary/40 transition-all duration-500 hover:-translate-y-2 hover:shadow-[0_30px_60px_rgba(78,222,163,0.15)] overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-br from-secondary/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
              <div className="absolute top-0 left-0 w-full h-[2px] bg-gradient-to-r from-transparent via-secondary/50 to-transparent transform scale-x-0 group-hover:scale-x-100 transition-transform duration-700" />

              {/* Highlight Glow bg */}
              <div className="absolute -bottom-20 -right-20 w-64 h-64 bg-secondary/15 rounded-full blur-[80px] opacity-0 group-hover:opacity-100 transition-opacity duration-700" />

              <div className="relative z-10 flex flex-col h-full gap-6">
                <div className="w-20 h-20 rounded-2xl bg-secondary/20 border border-secondary/30 flex items-center justify-center shadow-[0_0_20px_rgba(78,222,163,0.2)] group-hover:scale-110 transition-transform duration-500 ease-out">
                  <MaterialIcon icon="image_search" size={36} className="text-secondary" />
                </div>
                <div className="space-y-4 mt-auto">
                  <h2 className="text-3xl font-headline font-bold text-on-surface">Media Forensics Agent</h2>
                  <p className="text-on-surface-variant font-light text-lg">
                    Deep neural analysis on images, audio, and video files to verify authenticity and flag AI-manipulated content.
                  </p>
                </div>
              </div>
            </Link>
          </motion.div>
        </motion.div>
        
        {/* Footer controls or back button */}
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1, duration: 1 }}
          className="mt-16 flex items-center gap-12 text-sm font-label uppercase text-on-surface-variant tracking-[0.2em]"
        >
          <Link to="/login" className="hover:text-on-surface transition-colors cursor-pointer flex items-center gap-2">
            <MaterialIcon icon="login" size={16} /> Login
          </Link>
          <div className="w-1.5 h-1.5 rounded-full bg-outline-variant/30" />
          <Link to="/settings" className="hover:text-on-surface transition-colors cursor-pointer flex items-center gap-2">
            <MaterialIcon icon="settings" size={16} /> Device Setup
          </Link>
        </motion.div>
      </div>
    </div>
  );
};

export default Hub;
