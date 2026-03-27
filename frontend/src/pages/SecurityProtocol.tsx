import Layout from "@/components/Layout";
import MaterialIcon from "@/components/MaterialIcon";

const Section = ({ icon, title, children }: { icon: string; title: string; children: React.ReactNode }) => (
  <section className="space-y-5">
    <div className="flex items-center gap-4">
      <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center shrink-0">
        <MaterialIcon icon={icon} size={24} className="text-primary" />
      </div>
      <h2 className="text-2xl font-headline font-extrabold tracking-tight text-on-surface">{title}</h2>
    </div>
    <div className="pl-16 text-on-surface-variant text-base leading-relaxed space-y-4">{children}</div>
  </section>
);

const SecurityProtocol = () => (
  <Layout systemStatus="monitoring">
    <div className="pt-32 pb-24 px-6 md:px-16 max-w-5xl mx-auto">
      {/* Header */}
      <div className="mb-16">
        <div className="flex items-center gap-3 mb-4">
          <span className="h-px w-12 bg-primary" />
          <span className="text-primary font-label text-xs font-bold uppercase tracking-[0.190em]">Infrastructure</span> <span className="h-px w-12 bg-primary" />
        </div>
        <h1 className="font-headline text-4xl md:text-5xl font-extrabold tracking-tighter text-on-surface mb-5">
          Security Protocol<span className="text-primary-container">.</span>
        </h1>
        <p className="text-on-surface-variant text-lg max-w-3xl leading-relaxed">
          An overview of the state-of-the-art security measures protecting Satya Drishti's neural forensic pipeline and user data.
        </p>
        <div className="flex items-center gap-4 mt-6 text-xs text-on-surface-variant">
          <span className="px-3 py-1 rounded-full bg-surface-container-high border border-outline-variant/10 font-label uppercase tracking-widest">
            Framework: Zero Trust
          </span>
          <span className="px-3 py-1 rounded-full bg-surface-container-high border border-outline-variant/10 font-label uppercase tracking-widest">
            Last Updated: March 25, 2026
          </span>
        </div>
      </div>

      {/* Content */}
      <div className="space-y-14">
        <Section icon="verified" title="Zero Trust Architecture">
          <p>
            Satya Drishti employs a strict Zero Trust model. Every request, whether internal or external, must be authenticated, authorized, and continuously validated.
          </p>
          <ul className="list-disc pl-6 space-y-2 text-base">
            <li>Strict Identity and Access Management (IAM) controls</li>
            <li>Micro-segmentation of the backend neural network layers</li>
            <li>Least privilege principles applied to all automated services</li>
          </ul>
        </Section>

        <Section icon="enhanced_encryption" title="Cryptographic Standards">
          <p>
            All data interacting with the Satya Drishti platform is protected using modern cryptographic standards tailored for maximum security without sacrificing sub-second response times.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2">
            {[
              { icon: "vpn_key", text: "AES-256 for data at rest" },
              { icon: "lock", text: "TLS 1.3 for data in transit" },
              { icon: "password", text: "Argon2 / Bcrypt for password hashing" },
              { icon: "memory", text: "Hardened keys managed by HSMs" },
            ].map((item, i) => (
              <div key={i} className="flex items-center gap-4 p-4 rounded-lg bg-surface-container-high/50 border border-outline-variant/10">
                <MaterialIcon icon={item.icon} size={20} className="text-secondary shrink-0" />
                <span className="text-base">{item.text}</span>
              </div>
            ))}
          </div>
        </Section>

        <Section icon="developer_board" title="Infrastructure Protection">
          <p>
            Our API endpoints and forensic processing units are shielded against traditional and AI-driven attack vectors.
          </p>
          <ul className="list-disc pl-6 space-y-2 text-base">
            <li><strong>DDoS Mitigation:</strong> Enterprise-grade web application firewalls and traffic analyzers</li>
            <li><strong>Rate Limiting:</strong> Adaptive token bucket algorithms to prevent API abuse</li>
            <li><strong>Container Security:</strong> Immutable infrastructure with ephemeral containers</li>
          </ul>
        </Section>

        <Section icon="bug_report" title="Vulnerability Management">
          <p>
            We proactively identify and remediate security vulnerabilities in both our platform framework and our deep learning models.
          </p>
          <ul className="list-disc pl-6 space-y-2 text-base">
            <li>Continuous Static and Dynamic Application Security Testing (SAST/DAST)</li>
            <li>Adversarial robustness testing against our detection models</li>
            <li>Regular third-party penetration testing and audits</li>
          </ul>
        </Section>

        <Section icon="delete_forever" title="Volatile Data Processing">
          <p>
            Media uploads (audio, video, images) exist in highly isolated, ephemeral processing zones.
          </p>
          <p>
            Once the forensic pipeline generates a verdict (typically within 0.3 - 2 seconds), the source media buffer is overwritten and dropped from Volatile Memory (RAM). We do not serialize the media to a persistent disk.
          </p>
        </Section>
        
      </div>

      {/* Footer note */}
      <div className="mt-16 pt-8 border-t border-outline-variant/10 text-center">
        <p className="text-xs text-on-surface-variant/50 uppercase tracking-widest">
          &copy; 2026 Satya Drishti. All rights reserved. Built with privacy by design.
        </p>
      </div>
    </div>
  </Layout>
);

export default SecurityProtocol;
