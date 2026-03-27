import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import MaterialIcon from "@/components/MaterialIcon";
import TopBar from "@/components/TopBar";

const About = () => {
  return (
    <div className="min-h-screen bg-background text-on-surface overflow-x-hidden">
      <TopBar systemStatus="protected" />

      {/* Hero Section */}
      <section className="relative pt-40 pb-20 px-6 max-w-7xl mx-auto">
        <motion.div 
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="space-y-8"
        >
          <span className="inline-block px-4 py-1.5 bg-primary/10 border border-primary/20 rounded-full text-primary font-mono text-[10px] font-black uppercase tracking-[0.3em]">
            Institutional Forensics
          </span>
          <h1 className="text-6xl md:text-8xl font-black font-headline tracking-tighter leading-[0.85] uppercase">
            Defending Human <br /> <span className="text-primary-container">Integrity.</span>
          </h1>
          <p className="max-w-3xl text-xl md:text-2xl text-on-surface-variant font-light leading-relaxed">
            In an era where reality is software-defined, Satya Drishti serves as the definitive perimeter against synthetic media manipulation, voice cloning, and digital impersonation.
          </p>
        </motion.div>
      </section>

      {/* Core Values / Grid */}
      <section className="py-20 px-6 max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-8">
        {[
          {
            title: "Vision",
            icon: "visibility",
            desc: "To restore absolute trust in digital communication by making synthetic deception detectable, traceable, and irrelevant."
          },
          {
            title: "Technology",
            icon: "psychology",
            desc: "Powered by multi-modal neural artifact analysis, patch-based pixel statistics, and low-latency acoustic forensic fingerprinting."
          },
          {
            title: "Safety",
            icon: "verified_user",
            desc: "Built on a 'Zero Trust' media protocol, ensuring every frame and every sound bit is audited for authenticity before it reaches the human ear."
          }
        ].map((item, i) => (
          <motion.div 
            key={item.title}
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ delay: i * 0.1, duration: 0.5 }}
            className="p-10 rounded-[2.5rem] bg-surface-container-low/30 border border-white/5 backdrop-blur-xl hover:bg-surface-container-low/50 transition-all group"
          >
            <div className="w-16 h-16 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center mb-8 group-hover:bg-primary/20 group-hover:border-primary/40 transition-all">
              <MaterialIcon icon={item.icon} size={32} className="text-on-surface-variant group-hover:text-primary" />
            </div>
            <h3 className="text-3xl font-headline font-black mb-4 uppercase tracking-tight">{item.title}</h3>
            <p className="text-on-surface-variant text-lg font-light leading-snug">{item.desc}</p>
          </motion.div>
        ))}
      </section>

      {/* Detailed Story / Mission */}
      <section className="py-32 bg-surface-container-lowest/50 backdrop-blur-3xl border-y border-white/5">
        <div className="max-w-4xl mx-auto px-6 space-y-16">
          <div className="space-y-6">
             <h2 className="text-4xl md:text-5xl font-headline font-black uppercase tracking-tighter">The Deepfake Crisis</h2>
             <p className="text-xl text-on-surface-variant font-light leading-relaxed">
               Last year alone, synthetic media scams cost families over <span className="text-error font-bold italic">₹1,750 crore</span> in India. From voice-cloned ransom calls to AI-generated political misinformation, the weapons of deception are evolving faster than our collective ability to recognize them. 
             </p>
          </div>

          <div className="space-y-6">
             <h2 className="text-4xl md:text-5xl font-headline font-black uppercase tracking-tighter">Our Shield</h2>
             <p className="text-xl text-on-surface-variant font-light leading-relaxed">
               Satya Drishti was born from the necessity of total defense. We don't just 'guess' at authenticity—we perform a surgical dissection of media bits. Our pipeline looks for the phantom echoes of neural generation, the micro-inconsistencies in eye-glint patterns, and the synthetic tremors in cloned audio.
             </p>
          </div>
        </div>
      </section>

      <footer className="py-20 text-center text-outline text-[10px] font-mono uppercase tracking-[0.5em] opacity-30">
        Satya Drishti • Defending Humanity in the Age of AI
      </footer>
    </div>
  );
};

export default About;
