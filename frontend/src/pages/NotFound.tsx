import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import MaterialIcon from "@/components/MaterialIcon";

const NotFound = () => {
  return (
    <div className="min-h-screen bg-surface flex items-center justify-center relative overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0" aria-hidden="true">
        <div className="absolute top-1/3 left-1/3 w-[400px] h-[400px] rounded-full bg-primary/[0.04] blur-[120px]" />
        <div className="absolute bottom-1/3 right-1/3 w-[300px] h-[300px] rounded-full bg-secondary/[0.04] blur-[100px]" />
      </div>

      <motion.div
        className="text-center relative z-10 px-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="bg-surface-container-low rounded-xl border border-outline-variant/10 p-12 max-w-lg mx-auto space-y-6">
          <motion.div
            animate={{ y: [0, -8, 0] }}
            transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
            className="flex justify-center"
          >
            <MaterialIcon icon="security" size={64} className="text-primary/30" />
          </motion.div>

          <h1 className="font-headline text-4xl font-extrabold tracking-tighter text-on-surface">
            404
          </h1>

          <div className="space-y-2">
            <p className="text-xl text-on-surface-variant/60 font-headline">Page not found</p>
            <p className="text-sm text-on-surface-variant/40 max-w-md mx-auto">
              The page you're looking for doesn't exist or has been moved.
            </p>
          </div>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-3 pt-2">
            <Link
              to="/"
              className="inline-flex items-center gap-2 px-8 py-3.5 rounded-2xl bg-primary text-on-primary font-headline font-semibold hover:bg-primary/90 hover:scale-[1.02] transition-all cursor-pointer"
            >
              Return Home
            </Link>
            <Link
              to="/scanner"
              className="inline-flex items-center gap-2 px-8 py-3.5 rounded-2xl bg-surface-container-low border border-outline-variant/10 text-on-surface-variant font-medium hover:text-on-surface hover:border-outline-variant/20 transition-all cursor-pointer"
            >
              <MaterialIcon icon="search" size={16} />
              Go to Scanner
            </Link>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default NotFound;
