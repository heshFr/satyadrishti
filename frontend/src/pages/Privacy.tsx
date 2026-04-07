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

const Privacy = () => (
  <Layout systemStatus="monitoring">
    <div className="pt-32 pb-24 px-6 md:px-16 max-w-5xl mx-auto">
      {/* Header */}
      <div className="mb-16">
        <div className="flex items-center gap-3 mb-4">
          <span className="h-px w-12 bg-primary" />
          <span className="text-primary font-label text-xs font-bold uppercase tracking-[0.190em]">Legal</span> <span className="h-px w-12 bg-primary" />
        </div>
        <h1 className="font-headline text-4xl md:text-5xl font-extrabold tracking-tighter text-on-surface mb-5">
          Privacy Policy<span className="text-primary-container">.</span>
        </h1>
        <p className="text-on-surface-variant text-lg max-w-3xl leading-relaxed">
          Your privacy is fundamental to our mission. Satya Drishti is designed to protect truth, and that starts with protecting your data.
        </p>
        <div className="flex items-center gap-4 mt-6 text-xs text-on-surface-variant">
          <span className="px-3 py-1 rounded-full bg-surface-container-high border border-outline-variant/10 font-label uppercase tracking-widest">
            Effective: March 25, 2026
          </span>
          <span className="px-3 py-1 rounded-full bg-surface-container-high border border-outline-variant/10 font-label uppercase tracking-widest">
            Last Updated: April 6, 2026
          </span>
        </div>
      </div>

      {/* Content */}
      <div className="space-y-14">
        <Section icon="info" title="Introduction">
          <p>
            Satya Drishti ("we", "our", or "the Service") is a multimodal deepfake
            and coercion detection platform. This Privacy Policy explains how we collect, use, disclose, and safeguard
            your information when you use our web application, API services, and related tools.
          </p>
          <p>
            By accessing or using Satya Drishti, you agree to the terms described in this policy. If you do not agree,
            please discontinue use immediately.
          </p>
        </Section>

        <Section icon="database" title="Information We Collect">
          <p className="font-semibold text-on-surface">Account Information</p>
          <ul className="list-disc pl-6 space-y-2 text-base">
            <li>Email address and display name (during registration)</li>
            <li>Hashed password (never stored in plaintext)</li>
            <li>OAuth profile data if using Google or GitHub sign in (email, name only)</li>
            <li>Language and interface preferences</li>
          </ul>
          <p className="font-semibold text-on-surface pt-2">Media Uploads</p>
          <ul className="list-disc pl-6 space-y-2 text-base">
            <li>Images, videos, and audio files submitted for analysis</li>
            <li>Uploaded files are processed in memory and <strong>immediately deleted</strong> from the server after analysis completes</li>
            <li>We do not store, archive, or train on your uploaded media</li>
          </ul>
          <p className="font-semibold text-on-surface pt-2">Scan Results</p>
          <ul className="list-disc pl-6 space-y-2 text-base">
            <li>Forensic analysis verdicts, confidence scores, and forensic layer data</li>
            <li>Results are stored only if you are signed in and are associated with your account</li>
            <li>You may delete your scan history at any time</li>
          </ul>
        </Section>

        <Section icon="shield" title="How We Use Your Information">
          <ul className="list-disc pl-6 space-y-3 text-base">
            <li><strong>Analysis and Detection:</strong> To perform deepfake detection, coercion analysis, and media forensics on files you submit</li>
            <li><strong>Account Management:</strong> To authenticate your identity and maintain your preferences</li>
            <li><strong>Scan History:</strong> To let you review past analyses and generate evidence reports</li>
            <li><strong>Service Improvement:</strong> Aggregate, anonymized usage statistics may be used to improve detection accuracy (never individual media content)</li>
            <li><strong>Legal Compliance:</strong> To comply with applicable laws, regulations, or legal processes</li>
          </ul>
        </Section>

        <Section icon="visibility_off" title="Information We Never Collect">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {[
              "Browsing history or cookies for tracking",
              "Location data or GPS",
              "Device fingerprints or advertising IDs",
              "Contacts, call logs, or SMS",
              "Biometric templates (facial data is processed but never stored)",
              "Third party analytics or advertising scripts",
            ].map((item, i) => (
              <div key={i} className="flex items-start gap-2">
                <MaterialIcon icon="block" size={16} className="text-error mt-0.5 shrink-0" />
                <span className="text-base">{item}</span>
              </div>
            ))}
          </div>
        </Section>

        <Section icon="lock" title="Data Security">
          <p>We use strong security measures to protect your data:</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2">
            {[
              { icon: "encrypted", text: "AES-256 encryption at rest" },
              { icon: "https", text: "TLS 1.3 encryption in transit" },
              { icon: "password", text: "Bcrypt password hashing with salting" },
              { icon: "token", text: "JWT tokens with configurable expiry" },
              { icon: "speed", text: "Rate limiting on all API endpoints" },
              { icon: "delete_sweep", text: "Automatic cleanup of temporary files" },
            ].map((item, i) => (
              <div key={i} className="flex items-center gap-4 p-4 rounded-lg bg-surface-container-high/50 border border-outline-variant/10">
                <MaterialIcon icon={item.icon} size={20} className="text-secondary shrink-0" />
                <span className="text-base">{item.text}</span>
              </div>
            ))}
          </div>
        </Section>

        <Section icon="share" title="Data Sharing and Third Parties">
          <p>
            Satya Drishti does <strong>not</strong> sell, rent, or trade your personal information to third parties.
            We may share data only in these limited circumstances:
          </p>
          <ul className="list-disc pl-6 space-y-3 text-base">
            <li><strong>With your consent:</strong> When you explicitly choose to share evidence reports or case data</li>
            <li><strong>Legal obligations:</strong> When required by law, court order, or government regulation</li>
            <li><strong>Cyber crime reporting:</strong> If you voluntarily submit a case to law enforcement through our platform</li>
            <li><strong>OAuth providers:</strong> Google and GitHub receive only the minimum data required for authentication (no media content)</li>
          </ul>
        </Section>

        <Section icon="storage" title="Data Retention">
          <ul className="list-disc pl-6 space-y-3 text-base">
            <li><strong>Uploaded media:</strong> Deleted right after analysis (not stored on our servers)</li>
            <li><strong>Scan results:</strong> Retained until you delete them or delete your account</li>
            <li><strong>Account data:</strong> Kept for the lifetime of your account, deleted when you request account deletion</li>
            <li><strong>Server logs:</strong> Retained for a maximum of 30 days for security monitoring</li>
          </ul>
        </Section>

        <Section icon="manage_accounts" title="Your Rights">
          <p>You have the following rights regarding your personal data:</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2">
            {[
              { title: "Access", desc: "Request a copy of all data we hold about you" },
              { title: "Rectification", desc: "Update or correct your personal information" },
              { title: "Deletion", desc: "Delete your account and all associated data" },
              { title: "Portability", desc: "Export your scan history in CSV format" },
              { title: "Restriction", desc: "Request we limit processing of your data" },
              { title: "Objection", desc: "Object to any data processing activities" },
            ].map((item, i) => (
              <div key={i} className="p-4 rounded-lg bg-surface-container-high/50 border border-outline-variant/10">
                <p className="text-base font-bold text-on-surface">{item.title}</p>
                <p className="text-sm text-on-surface-variant mt-1">{item.desc}</p>
              </div>
            ))}
          </div>
          <p className="pt-2">
            To exercise any of these rights, contact us at{" "}
            <a href="mailto:privacy@satyadrishti.in" className="text-primary hover:underline">privacy@satyadrishti.in</a>.
          </p>
        </Section>

        <Section icon="child_care" title="Children's Privacy">
          <p>
            Satya Drishti is not intended for use by individuals under the age of 13. We do not knowingly collect
            personal information from children. If we learn that a child under 13 has provided us with personal data,
            we will take steps to delete it promptly.
          </p>
        </Section>

        <Section icon="cookie" title="Cookies and Local Storage">
          <p>
            We use <strong>only essential cookies and local storage</strong> for functionality:
          </p>
          <ul className="list-disc pl-6 space-y-2 text-base">
            <li><strong>Authentication token:</strong> Stored in localStorage to maintain your session</li>
            <li><strong>Language preference:</strong> Stored locally to remember your interface language</li>
            <li><strong>Theme preference:</strong> Stored locally for dark/light mode</li>
          </ul>
          <p>We do not use tracking cookies, analytics cookies, or advertising cookies.</p>
        </Section>

        <Section icon="gavel" title="Legal Basis (GDPR)">
          <p>For users in the European Economic Area, our legal bases for processing are:</p>
          <ul className="list-disc pl-6 space-y-2 text-base">
            <li><strong>Consent:</strong> You consent to analysis when you upload media</li>
            <li><strong>Contract:</strong> Processing necessary to provide the service you requested</li>
            <li><strong>Legitimate interest:</strong> Security monitoring and fraud prevention</li>
            <li><strong>Legal obligation:</strong> Compliance with applicable laws</li>
          </ul>
        </Section>

        <Section icon="public" title="India Digital Personal Data Protection Act (DPDP Act, 2023)">
          <p>For users in India, Satya Drishti complies with the Digital Personal Data Protection Act, 2023:</p>
          <ul className="list-disc pl-6 space-y-3 text-base">
            <li><strong>Lawful Purpose:</strong> We process personal data only for the purpose of providing deepfake and coercion detection services that you have explicitly consented to by using the platform</li>
            <li><strong>Data Minimization:</strong> We collect only the minimum data necessary to operate the service. Uploaded media is processed in memory and deleted immediately after analysis</li>
            <li><strong>Notice:</strong> This privacy policy serves as the required notice under Section 5 of the DPDP Act, clearly describing what data we collect, why, and how we process it</li>
            <li><strong>Consent:</strong> By creating an account and uploading media, you provide informed consent. You may withdraw consent at any time by deleting your account</li>
            <li><strong>Data Principal Rights:</strong> You have the right to access, correct, and erase your personal data. Use your account settings or contact us to exercise these rights</li>
            <li><strong>Grievance Redressal:</strong> Contact our Data Protection Officer at <a href="mailto:eksatyadrishti@gmail.com" className="text-primary hover:underline">eksatyadrishti@gmail.com</a> for any privacy concerns. We will respond within 30 days</li>
            <li><strong>Data Localization:</strong> Primary data processing occurs on servers accessible to Indian users. No personal data is transferred to jurisdictions without adequate data protection</li>
            <li><strong>Children:</strong> We do not process personal data of individuals below 18 years without verifiable parental consent, in compliance with Section 9 of the DPDP Act</li>
            <li><strong>Data Breach:</strong> In the event of a data breach, we will notify the Data Protection Board of India and affected users as required under Section 8(6)</li>
          </ul>
        </Section>

        <Section icon="edit_note" title="Changes to This Policy">
          <p>
            We may update this Privacy Policy from time to time. Changes will be posted on this page with an updated
            "Last Updated" date. Continued use of Satya Drishti after changes means you accept
            the revised policy.
          </p>
        </Section>

        <Section icon="mail" title="Contact Us">
          <p>If you have any questions about this Privacy Policy or our data practices, contact us:</p>
          <div className="p-4 rounded-xl bg-surface-container-high/50 border border-outline-variant/10 space-y-2">
            <p className="text-sm"><strong className="text-on-surface">Email:</strong>{" "}
              <a href="mailto:privacy@satyadrishti.in" className="text-primary hover:underline">eksatyadrishti@gmail.com</a>
            </p>
            <p className="text-sm"><strong className="text-on-surface">Project:</strong> Satya Drishti</p>
            <p className="text-sm"><strong className="text-on-surface">Cyber Crime Portal:</strong>{" "}
              <a href="https://cybercrime.gov.in" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">cybercrime.gov.in</a>
            </p>
          </div>
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

export default Privacy;
