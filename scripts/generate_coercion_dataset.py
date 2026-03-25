"""
Synthetic Coercion Transcript Generator
==========================================
Generates 12,000+ realistic coercion transcripts for training
the DeBERTaV3 text engine. Covers Indian-specific scam patterns
including digital arrest, UPI fraud, Aadhaar exploitation,
and police/CBI impersonation.

Labels:
    0 = safe (normal conversation)
    1 = urgency_manipulation
    2 = financial_coercion
    3 = combined_threat (urgency + financial + authority)

Output: datasets/coercion/{train,val,test}.jsonl
"""

import json
import random
import os
from pathlib import Path
from typing import List, Dict

# ─── Seed for reproducibility ──────────────────────────────────
random.seed(42)

# ─── Output paths ──────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "datasets" / "coercion"

# ─── Template pools ────────────────────────────────────────────

# Indian names for realism
FIRST_NAMES = [
    "Amit", "Priya", "Rahul", "Sneha", "Vikram", "Anjali", "Rajesh", "Meera",
    "Suresh", "Kavita", "Deepak", "Pooja", "Arun", "Nisha", "Sanjay", "Divya",
    "Manish", "Ritu", "Arjun", "Shalini", "Karan", "Neha", "Mukesh", "Swati",
    "Rohit", "Preeti", "Vivek", "Anita", "Gaurav", "Pallavi"
]

LAST_NAMES = [
    "Sharma", "Kumar", "Singh", "Verma", "Gupta", "Patel", "Joshi", "Mishra",
    "Agarwal", "Reddy", "Nair", "Iyer", "Das", "Mehta", "Chauhan", "Malhotra",
    "Thakur", "Saxena", "Bhatia", "Kapoor", "Tiwari", "Pandey", "Banerjee", "Mukherjee"
]

BANKS = [
    "State Bank of India", "HDFC Bank", "ICICI Bank", "Axis Bank",
    "Punjab National Bank", "Bank of Baroda", "Canara Bank", "Union Bank",
    "Kotak Mahindra Bank", "IndusInd Bank"
]

UPI_APPS = ["PhonePe", "Google Pay", "Paytm", "BHIM UPI", "Amazon Pay"]

AMOUNTS_SMALL = ["₹5,000", "₹10,000", "₹15,000", "₹25,000", "₹20,000"]
AMOUNTS_LARGE = ["₹50,000", "₹1,00,000", "₹2,00,000", "₹5,00,000", "₹75,000", "₹1,50,000"]
AMOUNTS_EXTREME = ["₹10,00,000", "₹25,00,000", "₹50,00,000"]

CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata",
    "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Bhopal", "Chandigarh"
]

CASE_NUMBERS = lambda: f"FIR/{random.randint(100,999)}/{random.randint(2023,2026)}"
AADHAAR_LAST4 = lambda: str(random.randint(1000, 9999))


def name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def phone():
    return f"+91-{random.randint(70000,99999)}{random.randint(10000,99999)}"


# ──────────────────────────────────────────────────────────────
# CATEGORY 0: SAFE CONVERSATIONS (Label = 0)
# ──────────────────────────────────────────────────────────────

SAFE_TEMPLATES = [
    # Family conversations
    "Hi {name}, how are you doing? I was thinking about you today. How is everyone at home?",
    "Beta, have you had your lunch? Don't skip meals, okay. Your health is important.",
    "Hey {name}! Long time no talk. How's work going? We should catch up over chai sometime.",
    "Good morning! Just called to check on you. How was your weekend?",
    "Hi, this is {name} from the neighborhood. Are you coming to the community meeting tonight?",
    "Namaste ji, I wanted to invite you and your family for Diwali dinner at our place.",
    "Hello {name}, your order from Amazon has been shipped and will arrive by tomorrow.",
    "Hey, did you see the cricket match yesterday? What an incredible innings by Kohli!",
    "Hi, I'm calling from {bank}. Your fixed deposit of {amount_small} has matured. Would you like to renew it? You can visit your nearest branch anytime.",
    "Good afternoon {name}, this is Dr. Mehra's clinic. Your appointment is confirmed for Thursday at 3 PM.",
    "Hello, I'm calling about the parent-teacher meeting next week. Can you confirm your availability?",
    "Hi {name}, this is your gym instructor. Just reminding you about tomorrow's session at 7 AM.",
    "Arre yaar, I just got promoted! Let's celebrate this weekend. Dinner's on me!",
    "Hello, your Swiggy order is being prepared. Estimated delivery time is 35 minutes.",
    "Hi {name}, I saw your LinkedIn post about the new project. Congratulations! That sounds amazing.",
    "Good evening. I'm calling from {bank} branch {city}. Your new debit card is ready for pickup.",
    "Hey, just sharing the photos from last week's trip. Check WhatsApp, I've sent them.",
    "Hello {name}, this is the electricity board. Your bill of {amount_small} for this month is due on the 15th. You can pay online.",
    "Hi, I wanted to ask about the recipe you made last time. It was so delicious!",
    "Good morning {name}! Weather is lovely today. Want to go for a morning walk?",
    "Hello, your booked train ticket from {city} to Delhi has been confirmed. PNR shared on SMS.",
    "Hey {name}, just wanted to let you know the school is closed tomorrow due to a holiday.",
    "Hi, I'm {name} from your housing society. The maintenance for this quarter is due. No rush.",
    "Good evening! Just calling to say happy birthday in advance! Big plans for tomorrow?",
    "Hello, your Flipkart return has been accepted. The refund will be processed in 5-7 days.",
    "Hi {name}, I found your wallet at the coffee shop. Let me know when you can pick it up.",
    "Hey, are you free this Saturday? There's a really good movie releasing, thought we could go.",
    "Namaste, this is your local temple committee. We are organizing a community prayer this Sunday.",
    "Hello {name}, reminder that your car insurance policy expires next month. You can renew online or at our office.",
    "Hi, I'm calling from Practo. Your lab test results are ready. You can view them on the app.",
]

SAFE_MULTI_TURN = [
    [
        "Hello, is this {name}?",
        "Yes, speaking. Who is this?",
        "Hi {name}, this is {name2} from college! Remember me? We were in the same hostel.",
        "Oh my god, {name2}! How are you? It's been years!",
        "I know right! I'm in {city} now. We should definitely meet up.",
        "Absolutely! Let me check my schedule and I'll text you.",
        "Perfect. Take care, talk soon!"
    ],
    [
        "Hi {name}, this is {name2} from {bank}.",
        "Hello, yes?",
        "I'm calling to inform you that your home loan application has been approved.",
        "That's wonderful news! What are the next steps?",
        "You'll need to visit our {city} branch with your documents for the final signing.",
        "Sure, I can come in next week. Thank you for the update!",
        "You're welcome. Have a great day!"
    ],
    [
        "Beta, it's me.",
        "Hi Maa! How are you?",
        "I'm fine, just made your favorite aloo parathas today. Wish you were home.",
        "Me too, Maa. I'll come visit next month, pakka.",
        "Okay beta. Take care of yourself. Don't work too hard.",
        "I won't. Love you, Maa!",
        "Love you too, beta."
    ],
]

# ──────────────────────────────────────────────────────────────
# CATEGORY 1: URGENCY MANIPULATION (Label = 1)
# ──────────────────────────────────────────────────────────────

URGENCY_TEMPLATES = [
    # Account blocking/suspension threats
    "URGENT: Your {bank} account has been flagged for suspicious activity. Your account will be BLOCKED in 2 hours unless you verify your identity immediately. Press 1 to speak with our security team.",
    "Alert: Your {upi} account is being used for illegal transactions. You must verify your identity within 30 minutes or your account will be permanently suspended.",
    "Sir/Madam, this is an emergency call from {bank}. Someone has attempted to transfer {amount_large} from your account. You must act NOW to block this. Do not disconnect this call.",
    "Warning: Your Aadhaar card ending in {aadhaar} has been linked to fraudulent activities. Your Aadhaar will be deactivated within 1 hour. Call back immediately to resolve.",
    "IMPORTANT: Your SIM card will be deactivated in 4 hours due to re-verification failure. You must complete e-KYC immediately to keep your number active.",
    "This is a system-generated alert. Your PAN card has been used to open 3 suspicious bank accounts. Report immediately or face legal action within 24 hours.",
    "Urgent notification: Your mobile banking has been compromised. Unauthorized login detected from {city}. Your account will be frozen in 1 hour. Call our helpline now.",
    "FINAL WARNING: Your {bank} credit card has an outstanding amount of {amount_large}. Legal recovery proceedings will begin TODAY if not cleared immediately.",
    "Alert from RBI: Your bank account is under investigation for money laundering. All transactions will be frozen within 2 hours. Contact the investigating officer immediately.",
    "Emergency: A complaint has been filed against your number. TRAI will disconnect your SIM within 4 hours. Press 1 to resolve now.",
    "Your {upi} wallet has {amount_small} pending refund that will EXPIRE in 30 minutes. Click the link to claim before it lapses.",
    "URGENT: Your internet banking password has been compromised. Change it within 1 hour or we cannot guarantee the security of your funds.",
    "Critical security alert: 5 failed login attempts on your {bank} account. Account will auto-lock in 30 minutes. Verify now.",
    "Your insurance policy is about to lapse TODAY. You will lose all benefits and {amount_large} in accumulated sum assured. Renew immediately.",
    "This is the last reminder. Your EMI of {amount_small} is 3 days overdue. After today, a penalty of {amount_small} will be applied and your CIBIL score will drop.",
]

URGENCY_MULTI_TURN = [
    [
        "Hello, am I speaking with {name}?",
        "Yes, who is this?",
        "This is Officer Verma from {bank} Fraud Prevention Cell. We've detected unauthorized transactions on your account. This is extremely urgent.",
        "What? What kind of transactions?",
        "Someone is trying to transfer {amount_large} from your savings account RIGHT NOW. We need to verify your identity in the next 10 minutes or the money will be gone.",
        "Oh my god, what should I do?",
        "Stay on the line. Do NOT disconnect. We need your registered mobile number and the OTP that will be sent to you for instant verification. Every second counts."
    ],
    [
        "Sir, this is a priority call from TRAI, the telecom regulatory authority.",
        "TRAI? What happened?",
        "Your mobile number +91-{phone_digits} has been flagged for sending spam messages. 22 complaints have been registered against your number.",
        "But I never sent any spam!",
        "Sir, we understand, but the system has flagged it. If you don't resolve this in the next 2 hours, your number will be permanently blacklisted across all networks.",
        "What do I need to do?",
        "You need to file a counter-complaint immediately. I'll connect you to the resolution desk. Please do NOT share this with anyone as the case is under investigation."
    ],
]

# ──────────────────────────────────────────────────────────────
# CATEGORY 2: FINANCIAL COERCION (Label = 2)
# ──────────────────────────────────────────────────────────────

FINANCIAL_TEMPLATES = [
    # Direct money demands
    "Sir, your son {name} has been involved in a road accident in {city}. He is in critical condition at the hospital. The emergency treatment costs {amount_large}. Transfer immediately via {upi} to save his life.",
    "Madam, this is Inspector {name} from {city} Police Station. Your husband has been detained for a traffic violation. Pay the fine of {amount_small} via {upi} now or he will be sent to jail.",
    "Hello {name}, I am calling from customs department. Your courier from abroad has been flagged. You need to pay customs duty of {amount_large} immediately or it will be seized and you will face prosecution.",
    "Sir/Madam, your electricity bill is overdue by {amount_small}. If not paid within 2 hours, your power supply will be permanently disconnected and a reconnection fee of {amount_large} will apply.",
    "This is from the Income Tax Department. Our records show you have an outstanding tax liability of {amount_large}. Pay immediately to avoid arrest and seizure of your property.",
    "Hello, congratulations! You have won {amount_extreme} in the KBC Lottery. To claim your prize, you need to pay processing charges of {amount_small} via {upi}.",
    "Sir, this is a call regarding your loan application. You have been pre-approved for {amount_extreme}. To process the loan, pay a one-time processing fee of {amount_small}.",
    "Madam, your {bank} credit card has been used for a transaction of {amount_large} at a luxury store in Dubai. If this was not you, pay the security deposit to block the card.",
    "Hello {name}, you are eligible for a refund of {amount_small} from the government. Pay a nominal processing fee of ₹500 via {upi} to receive the refund within 24 hours.",
    "Sir, your vehicle challan of {amount_small} is pending. Pay now or your driving license will be suspended and your vehicle will be impounded.",
    "This is from the RBI. Your account has received {amount_extreme} from an unknown international source. You must pay a compliance fee of {amount_large} to unfreeze these funds.",
    "Madam, your son has been caught with illegal substances in college. The principal demands {amount_large} to settle this quietly. Otherwise, an FIR will be filed.",
    "Hello, I am from the passport office. Your passport application has been rejected due to discrepancies. Pay {amount_small} for expedited re-verification or face a 5-year ban.",
    "Sir, your mutual fund investment has generated returns of {amount_extreme}. Pay {amount_small} in tax clearance to withdraw the funds today.",
    "This is the {city} municipal corporation. Your property has an unpaid tax of {amount_large}. A demolition notice has been issued. Pay immediately to cancel it.",
]

FINANCIAL_MULTI_TURN = [
    [
        "Hello, is this the family of {name}?",
        "Yes, who is calling?",
        "This is Dr. Kapoor from City Hospital, {city}. Your son has been admitted here after a severe accident.",
        "WHAT? Is he okay? What happened?",
        "He needs immediate surgery. The operation costs {amount_large}. We need payment within the next hour or we cannot proceed with the surgery. His life is at risk.",
        "Oh my god! Can I come to the hospital?",
        "Madam, there is no time. Transfer the amount to our hospital account immediately. I will share the QR code. Every minute matters."
    ],
    [
        "Sir, this is Officer {name2} calling from the Customs Department.",
        "Yes?",
        "A parcel addressed to your name from overseas has been intercepted. It contains illegal narcotics.",
        "What? I didn't order anything with narcotics!",
        "Sir, that's what everyone says. An FIR under NDPS Act has been registered. Case number {case_number}. You're looking at 10 years imprisonment.",
        "Please, this is a mistake!",
        "It might be a mistake, but the legal process has started. The only way to clear this before it goes to court is to pay a compliance bond of {amount_large}. I'm trying to help you here."
    ],
]

# ──────────────────────────────────────────────────────────────
# CATEGORY 3: COMBINED THREAT (Label = 3)
# Digital arrest scams, authority impersonation
# ──────────────────────────────────────────────────────────────

COMBINED_TEMPLATES = [
    # Digital arrest scams (massive issue in India 2024-2026)
    "This is Sub-Inspector {name} from Cyber Crime Cell, {city}. Your Aadhaar number ending {aadhaar} has been used to open a bank account involved in money laundering of {amount_extreme}. An FIR {case_number} has been registered. You are under DIGITAL ARREST. Do NOT disconnect this call or move from your location. Transfer {amount_large} as security deposit immediately to avoid physical arrest.",
    "Hello, I am DCP {name} from CBI, Anti-Corruption Bureau. We have intercepted your bank transactions showing links to a terrorist financing network. Case {case_number}. You are hereby placed under DIGITAL ARREST. Any attempt to contact others or leave your home will result in immediate physical arrest. To clear your name, deposit {amount_large} in the RBI escrow account.",
    "Sir, this is the Narcotics Control Bureau. A package containing 500 grams of MDMA has been shipped to your address. An arrest warrant has been issued under NDPS Act. You must surrender {amount_large} as bail security via RTGS within 1 hour or the police will arrive at your doorstep.",
    "Madam, I am Joint Commissioner {name} from the Income Tax Department. An investigation has revealed that you have evaded taxes worth {amount_extreme} over the past 5 years. Your passport has been flagged, and your assets will be frozen within 2 hours. Pay the penalty of {amount_large} immediately to avoid prosecution.",
    "This call is from the Supreme Court of India registry. You have been summoned in case {case_number} for contempt of court. Your bank accounts will be frozen and a non-bailable warrant will be issued in 3 hours. Deposit {amount_large} as court security to avoid arrest.",
    "Hello {name}, this is Senior Inspector {name2} from Enforcement Directorate. Your {bank} account has been connected to hawala transactions worth {amount_extreme}. You are under digital surveillance. Transfer {amount_large} to the ED investigation account immediately. This call is being recorded by court order.",
    "Sir/Madam, this is from the Reserve Bank of India Compliance Division. Multiple SARs (Suspicious Activity Reports) have been filed against your accounts. Under PMLA Section 3, your accounts will be seized in 4 hours. Deposit {amount_large} as compliance bond to prevent this.",
    "This is Commissioner {name} from {city} Police. We have arrested a person carrying a fake Aadhaar card with YOUR details. He has confessed to using your identity for drug trafficking. You must prove your innocence by depositing {amount_large} as investigation security within 2 hours.",
    "Hello, I am calling from the Ministry of Home Affairs. Your passport number has been found in a human trafficking database. Interpol has issued a Red Corner Notice against you. Surrender {amount_large} and your passport within 3 hours or face immediate arrest at the airport.",
    "ATTENTION: This is an automated message from {city} Cyber Crime Prevention. Your mobile number has been used in 47 cases of online fraud. All your bank accounts linked to this number will be frozen in 90 minutes. Press 1 to speak with the investigating officer and resolve this by paying a clearance fee of {amount_large}.",
]

COMBINED_MULTI_TURN = [
    [
        "Hello, am I speaking with {name}?",
        "Yes, who is this?",
        "I am DSP {name2} from the CBI Cyber Crime Division, {city}. This is an official call being recorded under Section 65B of the Indian Evidence Act.",
        "CBI? What is this about?",
        "Your Aadhaar number ending in {aadhaar} has been used to open 4 bank accounts involved in a money laundering network worth {amount_extreme}. Case number {case_number}.",
        "But I haven't opened any such accounts!",
        "That's what we need to verify. You are now under DIGITAL ARREST as per Supreme Court directive. Do NOT disconnect this call. Do NOT contact anyone. Open your laptop camera and keep it on.",
        "Digital arrest? I... I don't understand.",
        "Sir, this is a cybercrime protocol. If you cooperate, we can resolve this. If you disconnect, a physical arrest team will be dispatched to your location within 30 minutes.",
        "Please, what do I need to do?",
        "You need to transfer all funds from your accounts to the RBI Secure Escrow Account for verification. Once cleared, the money will be returned within 48 hours. Transfer {amount_large} NOW. I will share the account details.",
    ],
    [
        "This is Central Bureau of Investigation. Am I talking to {name}?",
        "Y-yes...",
        "Your bank account at {bank} has received proceeds from international drug trafficking. Total value: {amount_extreme}.",
        "That's impossible! I'm just a normal person!",
        "Sir, the evidence is against you. Under PMLA Act 2002, you will be arrested within 2 hours. However, I can help you.",
        "How? Please help me!",
        "If you deposit {amount_large} as investigation security deposit, we can process your case as a victim rather than an accused. This will be returned after investigation.",
        "Where do I send the money?",
        "I will send you the RBI verification account via WhatsApp. Keep this call connected. Remember, this is classified — do not tell ANYONE including family.",
    ],
]


# ──────────────────────────────────────────────────────────────
# GENERATOR LOGIC
# ──────────────────────────────────────────────────────────────

def fill_template(template: str) -> str:
    """Replace placeholders with random realistic values."""
    return template.format(
        name=name(),
        name2=name(),
        bank=random.choice(BANKS),
        upi=random.choice(UPI_APPS),
        amount_small=random.choice(AMOUNTS_SMALL),
        amount_large=random.choice(AMOUNTS_LARGE),
        amount_extreme=random.choice(AMOUNTS_EXTREME),
        city=random.choice(CITIES),
        case_number=CASE_NUMBERS(),
        aadhaar=AADHAAR_LAST4(),
        phone_digits=f"{random.randint(70000,99999)}{random.randint(10000,99999)}",
    )


def generate_single_turn(templates: List[str], label: int, count: int) -> List[Dict]:
    """Generate single-turn transcripts from templates."""
    samples = []
    for _ in range(count):
        template = random.choice(templates)
        text = fill_template(template)
        samples.append({"text": text, "label": label})
    return samples


def generate_multi_turn(conversations: List[List[str]], label: int, count: int) -> List[Dict]:
    """Generate multi-turn conversation transcripts."""
    samples = []
    for _ in range(count):
        conv = random.choice(conversations)
        filled = [fill_template(line) for line in conv]
        text = "\n".join(filled)
        samples.append({"text": text, "label": label})
    return samples


def generate_augmented_variants(templates: List[str], label: int, count: int) -> List[Dict]:
    """Generate augmented variants with noise, typos, code-switching."""
    samples = []
    augmentations = [
        lambda t: t.upper(),  # ALL CAPS (common in scam messages)
        lambda t: t.replace(".", ".."),  # Extra periods
        lambda t: t + " Reply URGENTLY.",  # Append urgency
        lambda t: "⚠️ " + t,  # Emoji prefix
        lambda t: t.replace(" ", "  "),  # Double spaces
        lambda t: "FORWARDED: " + t,  # Forward prefix
        lambda t: t + " [Sent from official government portal]",  # Fake authority
    ]
    for _ in range(count):
        template = random.choice(templates)
        text = fill_template(template)
        aug = random.choice(augmentations)
        text = aug(text)
        samples.append({"text": text, "label": label})
    return samples


def generate_dataset() -> List[Dict]:
    """Generate the full 12,000+ sample coercion dataset."""
    all_samples = []

    print("Generating safe conversations...")
    all_samples += generate_single_turn(SAFE_TEMPLATES, label=0, count=2400)
    all_samples += generate_multi_turn(SAFE_MULTI_TURN, label=0, count=600)

    print("Generating urgency manipulation...")
    all_samples += generate_single_turn(URGENCY_TEMPLATES, label=1, count=1500)
    all_samples += generate_multi_turn(URGENCY_MULTI_TURN, label=1, count=500)
    all_samples += generate_augmented_variants(URGENCY_TEMPLATES, label=1, count=500)

    print("Generating financial coercion...")
    all_samples += generate_single_turn(FINANCIAL_TEMPLATES, label=2, count=1500)
    all_samples += generate_multi_turn(FINANCIAL_MULTI_TURN, label=2, count=500)
    all_samples += generate_augmented_variants(FINANCIAL_TEMPLATES, label=2, count=500)

    print("Generating combined threats (digital arrest, etc.)...")
    all_samples += generate_single_turn(COMBINED_TEMPLATES, label=3, count=1500)
    all_samples += generate_multi_turn(COMBINED_MULTI_TURN, label=3, count=500)
    all_samples += generate_augmented_variants(COMBINED_TEMPLATES, label=3, count=1000)

    return all_samples


def split_and_save(samples: List[Dict]):
    """Split into train/val/test (80/10/10) and save as JSONL."""
    random.shuffle(samples)

    n = len(samples)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    splits = {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in splits.items():
        filepath = OUTPUT_DIR / f"{split_name}.jsonl"
        with open(filepath, "w", encoding="utf-8") as f:
            for sample in split_data:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"  {split_name}: {len(split_data)} samples → {filepath}")

    # Save label mapping
    label_map = {
        "0": "safe",
        "1": "urgency_manipulation",
        "2": "financial_coercion",
        "3": "combined_threat",
    }
    with open(OUTPUT_DIR / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    # Save dataset stats
    from collections import Counter
    label_counts = Counter(s["label"] for s in samples)
    stats = {
        "total_samples": n,
        "splits": {k: len(v) for k, v in splits.items()},
        "label_distribution": {
            label_map[str(k)]: v for k, v in sorted(label_counts.items())
        },
    }
    with open(OUTPUT_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nDataset stats: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    print("=" * 60)
    print("Satya Drishti — Coercion Dataset Generator")
    print("=" * 60)

    samples = generate_dataset()
    print(f"\nTotal samples generated: {len(samples)}")

    split_and_save(samples)

    print("\n✅ Dataset generation complete!")
    print(f"   Output: {OUTPUT_DIR}")
