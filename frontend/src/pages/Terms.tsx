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
            Last Updated: April 6, 2026
          </span>
        </div>
      </div>

      {/* Content */}
      <div className="space-y-14">
        <Section icon="gavel" title="1. Acceptance of Terms">
          <p>
            By accessing or using Satya Drishti, you agree to comply with and be bound by these Terms of Service. If you disagree with any part of these terms, you may not access the service.
          </p>
          <p>
            These terms apply to all users, including visitors, registered users, and API consumers.
          </p>
        </Section>

        <Section icon="verified_user" title="2. User Accounts">
          <p>
            You are responsible for safeguarding the password that you use to access the service and for any activities or actions under your password. You agree not to disclose your password to any third party.
          </p>
          <p>
            You must notify us immediately upon becoming aware of any breach of security or unauthorized use of your account.
          </p>
          <p>
            We reserve the right to suspend or terminate accounts that violate these terms, exhibit abusive usage patterns, or pose a security risk to the platform.
          </p>
        </Section>

        <Section icon="cloud_upload" title="3. Acceptable Use Policy">
          <p>
            You agree not to use the service for any purpose that is prohibited by these terms or by law. Specifically:
          </p>
          <ul className="list-disc pl-6 space-y-2 text-base">
            <li>You may not use the platform to harass, abuse, stalk, or harm another person</li>
            <li>You may not upload media that violates intellectual property rights or contains child sexual abuse material (CSAM)</li>
            <li>Reverse engineering, decompiling, or attempting to extract the detection algorithms, model weights, or proprietary techniques is strictly prohibited</li>
            <li>You may not use bots or automated scripts to mass-upload media without prior API authorization</li>
            <li>You may not attempt to circumvent rate limits, authentication mechanisms, or security controls</li>
            <li>You may not use analysis results to create deepfakes, improve deepfake generation, or evade detection</li>
            <li>You may not resell, redistribute, or sublicense access to the service without written permission</li>
            <li>You may not use the service to generate false evidence or fraudulent claims about media authenticity</li>
          </ul>
        </Section>

        <Section icon="api" title="4. API Terms for Developers">
          <p>
            The following additional terms apply to developers using the Satya Drishti API:
          </p>
          <ul className="list-disc pl-6 space-y-3 text-base">
            <li><strong>API Keys:</strong> You are responsible for securing your API keys. Do not embed keys in client-side code, public repositories, or shared environments. Compromised keys should be revoked immediately</li>
            <li><strong>Rate Limits:</strong> API access is subject to per-endpoint rate limits. Exceeding limits will result in temporary throttling (HTTP 429). Persistent abuse may result in key revocation</li>
            <li><strong>Fair Use:</strong> Free-tier API access is limited to 100 requests per day per key. Commercial usage requires a separate agreement</li>
            <li><strong>Attribution:</strong> Applications using the Satya Drishti API must display: "Powered by Satya Drishti" with a link back to our platform</li>
            <li><strong>No Caching of Verdicts:</strong> You may cache analysis results for display purposes but must not cache and redistribute verdicts as authoritative determinations without re-analysis</li>
            <li><strong>Data Handling:</strong> Media files sent via the API are subject to the same privacy policy as web uploads. Files are processed and immediately deleted</li>
            <li><strong>Uptime:</strong> We target 99% uptime but do not guarantee availability. The API is provided "as is" without SLA commitments on the free tier</li>
          </ul>
        </Section>

        <Section icon="copyright" title="5. Intellectual Property">
          <p>
            The service and its original content (excluding content provided by users), features, and functionality are and will remain the exclusive property of Satya Drishti and its licensors. This includes all detection algorithms, trained models, and analysis techniques.
          </p>
          <p>
            You retain full ownership of any media you upload. By uploading content to our platform for scanning, you grant us only the temporary right to process the media to determine its authenticity. We do not claim any ownership over your uploaded content.
          </p>
        </Section>

        <Section icon="shield" title="6. Content Policy">
          <p>Satya Drishti commits to the following content principles:</p>
          <ul className="list-disc pl-6 space-y-3 text-base">
            <li><strong>No Model Training:</strong> Uploaded media content is never used to train, fine-tune, or improve our detection models. Our models are trained exclusively on publicly available research datasets</li>
            <li><strong>No Third-Party Sharing:</strong> Analysis results and uploaded media are never shared with, sold to, or made available to third parties, except when required by law or court order</li>
            <li><strong>Right to Erasure:</strong> You may permanently delete your account and all associated data (scan history, cases, API keys) at any time through the account settings. This action is irreversible</li>
            <li><strong>Transparency:</strong> Our forensic reports detail exactly which detection layers were used and their individual assessments, providing full transparency into how verdicts are reached</li>
          </ul>
        </Section>

        <Section icon="warning" title="7. Detection Advisory Disclaimer">
          <div className="p-5 rounded-xl bg-error/5 border border-error/20 space-y-3">
            <p className="text-on-surface font-semibold">
              Satya Drishti's analysis results are advisory in nature and should not be treated as definitive proof of authenticity or manipulation.
            </p>
            <p>
              Our detection system uses state-of-the-art machine learning models, but no AI system achieves 100% accuracy. Results may include false positives (authentic media flagged as synthetic) or false negatives (manipulated media classified as authentic).
            </p>
            <p>
              <strong>You should not:</strong>
            </p>
            <ul className="list-disc pl-6 space-y-1">
              <li>Use our verdicts as the sole basis for legal proceedings or criminal complaints</li>
              <li>Treat a "deepfake" verdict as conclusive proof of manipulation without corroborating evidence</li>
              <li>Treat an "authentic" verdict as conclusive proof that media has not been tampered with</li>
              <li>Make financial, legal, or personal decisions based solely on our analysis</li>
            </ul>
            <p>
              For legal or investigative purposes, we recommend combining our analysis with expert forensic review. Our reports can serve as a preliminary screening tool to identify media that warrants further investigation.
            </p>
          </div>
        </Section>

        <Section icon="balance" title="8. Limitation of Liability">
          <p>
            In no event shall Satya Drishti, nor its directors, employees, partners, agents, suppliers, or affiliates, be liable for any indirect, incidental, special, consequential or punitive damages, including without limitation, loss of profits, data, use, goodwill, or other intangible losses, resulting from:
          </p>
          <ul className="list-disc pl-6 space-y-2 text-base">
            <li>Your access to or use of or inability to access or use the service</li>
            <li>Any incorrect detection results, including false positives or false negatives</li>
            <li>Actions taken based on our analysis results</li>
            <li>Unauthorized access to or alteration of your data</li>
            <li>Any third-party conduct on the service</li>
          </ul>
          <p>
            Our total aggregate liability shall not exceed the amount you have paid us in the twelve (12) months preceding the claim, or INR 1,000, whichever is greater.
          </p>
        </Section>

        <Section icon="security" title="9. Indemnification">
          <p>
            You agree to defend, indemnify, and hold harmless Satya Drishti from any claims, damages, obligations, losses, or expenses arising from your use of the service, your violation of these terms, or your violation of any rights of a third party.
          </p>
        </Section>

        <Section icon="public" title="10. Governing Law">
          <p>
            These Terms shall be governed by and construed in accordance with the laws of India, without regard to its conflict of law provisions. Any disputes shall be subject to the exclusive jurisdiction of the courts in Maharashtra, India.
          </p>
          <p>
            For users in the European Union, nothing in these terms affects your rights under mandatory EU consumer protection laws.
          </p>
        </Section>

        <Section icon="edit_note" title="11. Changes to Terms">
          <p>
            We reserve the right, at our sole discretion, to modify or replace these Terms at any time. If a revision is material we will try to provide at least 30 days' notice prior to any new terms taking effect. What constitutes a material change will be determined at our sole discretion.
          </p>
          <p>
            Continued use of the service after changes constitutes acceptance of the new terms.
          </p>
        </Section>

        <Section icon="mail" title="Contact Us">
          <p>If you have any questions about these Terms, please contact us:</p>
          <div className="p-4 rounded-xl bg-surface-container-high/50 border border-outline-variant/10 space-y-2">
            <p className="text-sm"><strong className="text-on-surface">Email:</strong>{" "}
              <a href="mailto:legal@satyadrishti.in" className="text-primary hover:underline">eksatyadrishti@gmail.com</a>
            </p>
            <p className="text-sm"><strong className="text-on-surface">Project:</strong> Satya Drishti</p>
            <p className="text-sm"><strong className="text-on-surface">Jurisdiction:</strong> Maharashtra, India</p>
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
