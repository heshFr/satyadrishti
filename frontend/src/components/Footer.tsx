import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";

const Footer = () => {
  const { t } = useTranslation();

  return (
    <footer className="w-full py-16 px-8 bg-surface-container-lowest border-t border-outline-variant/5 relative z-10">
      <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
        <div>
          <div className="text-lg font-black text-on-surface font-headline uppercase tracking-widest mb-4">
            Satya Drishti
          </div>
          <div className="text-xs tracking-widest uppercase text-on-surface-variant/60 font-body">
            {t("footer.copyright")}
          </div>
        </div>
        <div className="flex flex-wrap justify-start md:justify-end gap-8">
          <Link
            to="/help"
            className="text-on-surface-variant/60 hover:text-primary-container transition-colors font-body text-xs tracking-widest uppercase"
          >
            {t("common.help")}
          </Link>
          <Link
            to="/contact"
            className="text-on-surface-variant/60 hover:text-primary-container transition-colors font-body text-xs tracking-widest uppercase"
          >
            {t("common.contact")}
          </Link>
          <Link
            to="/donate"
            className="text-primary-container/80 hover:text-primary-container transition-colors font-body text-xs tracking-widest uppercase font-semibold"
          >
            {t("common.donate")}
          </Link>
          <a
            href="https://cybercrime.gov.in"
            target="_blank"
            rel="noopener noreferrer"
            className="text-on-surface-variant/60 hover:text-primary-container transition-colors font-body text-xs tracking-widest uppercase"
          >
            {t("footer.cyberCrimePortal")}
          </a>
          <Link to="/privacy" className="text-on-surface-variant/60 hover:text-primary-container transition-colors font-body text-xs tracking-widest uppercase">
            {t("footer.privacyPolicy")}
          </Link>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
