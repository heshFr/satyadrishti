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

const Terms = () => (
  <Layout systemStatus="monitoring">
    <div className="pt-32 pb-24 px-6 md:px-16 max-w-5xl mx-auto">
      {/* Header */}
      <div className="mb-16">
        <div className="flex items-center gap-3 mb-4">
          <span className="h-px w-12 bg-primary" />
          <span className="text-primary font-label text-xs font-bold uppercase tracking-[0.190em]">Legal</span> <span className="h-px w-12 bg-primary" />
        </div>
        <h1 className="font-headline text-4xl md:text-5xl font-extrabold tracking-tighter text-on-surface mb-5">
          Terms of Service<span className="text-primary-container">.</span>
        </h1>
        <p className="text-on-surface-variant text-lg max-w-3xl leading-relaxed">
          These Terms of Service govern your use of the Satya Drishti platform. By accessing or using our services, you agree to be bound by these terms.
        </p>
        <div className="flex items-center gap-4 mt-6 text-xs text-on-surface-variant">
          <span className="px-3 py-1 rounded-full bg-surface-container-high border border-outline-variant/10 font-label uppercase tracking-widest">
            Effective: March 25, 2026
          </span>
          <span className="px-3 py-1 rounded-full bg-surface-container-high border border-outline-variant/10 font-label uppercase tracking-widest">
            Last Updated: March 25, 2026
          </span>
        </div>
      </div>

      {/* Content */}
      <div className="space-y-14">
        <Section icon="gavel" title="1. Acceptance of Terms">
          <p>
            By accessing or using Satya Drishti, you agree to comply with and be bound by these Terms of Service. If you disagree with any part of these terms, you may not access the service.
          </p>
        </Section>

        <Section icon="verified_user" title="2. User Accounts">
          <p>
            You are responsible for safeguarding the password that you use to access the service and for any activities or actions under your password. You agree not to disclose your password to any third party.
          </p>
          <p>
            You must notify us immediately upon becoming aware of any breach of security or unauthorized use of your account.
          </p>
        </Section>

        <Section icon="cloud_upload" title="3. Acceptable Use">
          <p>
            You agree not to use the service for any purpose that is prohibited by these terms or by law.
          </p>
          <ul className="list-disc pl-6 space-y-2 text-base">
            <li>You may not use the platform to harass, abuse, or harm another person</li>
            <li>You may not upload media that violates intellectually property rights</li>
            <li>Reverse engineering the detection algorithms is strictly prohibited</li>
            <li>You may not use bots or automated scripts to mass-upload media without authorization</li>
          </ul>
        </Section>

        <Section icon="copyright" title="4. Intellectual Property">
          <p>
            The service and its original content (excluding content provided by users), features, and functionality are and will remain the exclusive property of Satya Drishti and its licensors.
          </p>
          <p>
            You retain full ownership of any media you upload. By uploading content to our platform for scanning, you grant us only the temporary right to process the media to determine its authenticity.
          </p>
        </Section>

        <Section icon="warning" title="5. Limitation of Liability">
          <p>
            In no event shall Satya Drishti, nor its directors, employees, partners, agents, suppliers, or affiliates, be liable for any indirect, incidental, special, consequential or punitive damages, including without limitation, loss of profits, data, use, goodwill, or other intangible losses, resulting from your access to or use of or inability to access or use the service.
          </p>
        </Section>

        <Section icon="balance" title="6. Disclaimer of Warranties">
          <p>
            Satya Drishti's deepfake detection models are provided on an "AS IS" and "AS AVAILABLE" basis. The service is provided without warranties of any kind, whether express or implied.
          </p>
          <p>
            While we strive for maximum accuracy, no detection algorithm is perfect. Our scan results constitute informed algorithmic opinions rather than absolute statements of fact.
          </p>
        </Section>

        <Section icon="edit_note" title="7. Changes to Terms">
          <p>
            We reserve the right, at our sole discretion, to modify or replace these Terms at any time. If a revision is material we will try to provide at least 30 days' notice prior to any new terms taking effect.
          </p>
        </Section>

        <Section icon="mail" title="Contact Us">
          <p>If you have any questions about these Terms, please contact us:</p>
          <div className="p-4 rounded-xl bg-surface-container-high/50 border border-outline-variant/10 space-y-2">
            <p className="text-sm"><strong className="text-on-surface">Email:</strong>{" "}
              <a href="mailto:legal@satyadrishti.in" className="text-primary hover:underline">legal@satyadrishti.in</a>
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

export default Terms;
