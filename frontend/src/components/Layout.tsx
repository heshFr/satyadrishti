import { ReactNode } from "react";
import TopBar from "./TopBar";
import Footer from "./Footer";
import { motion, AnimatePresence } from "framer-motion";
import { useLocation } from "react-router-dom";

interface LayoutProps {
  children: ReactNode;
  systemStatus: "protected" | "monitoring" | "alert";
}

const Layout = ({ children, systemStatus }: LayoutProps) => {
  const location = useLocation();

  return (
    <div className="min-h-screen bg-surface flex flex-col relative overflow-hidden">
      <TopBar systemStatus={systemStatus} />

      <AnimatePresence mode="wait">
        <motion.main
          key={location.pathname}
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -12 }}
          transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
          className="flex-1 relative z-10 pt-20"
        >
          {children}
        </motion.main>
      </AnimatePresence>

      <Footer />
    </div>
  );
};

export default Layout;
