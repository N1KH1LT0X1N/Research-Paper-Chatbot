# Research-Paper-Chatbot — WhatsApp Bot (Gemini)

Lightweight Flask-based WhatsApp assistant for semantic search, Q&A and concise summaries of research papers. The bot uses Twilio for WhatsApp messaging and Google Gemini for generation.

## Live demo (deployed)

- URL: https://research-paper-chatbot-2.onrender.com

## Use / Join

- WhatsApp: https://wa.me/14155238886?text=join%20pocket-afternoon

---

## Requirements

- Python 3.9+
- Twilio account (WhatsApp sandbox or approved number)
- Gemini API key (Generative AI API)

## Setup

1. Create a `.env` in `whatsapp-bot/` (or at repo root if you run the scripts directly):

```
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_twilio_auth_token
GEMINI_API_KEY=your_gemini_api_key
```

2. Install dependencies:

```
pip install -r whatsapp-bot/requirements.txt
```

3. Run the demo bot (Explain/Activities):

```
python -u whatsapp-bot/bot_demo.py
```

4. Run the Research Paper bot (search + Q&A):

```
python -u whatsapp-bot/research_bot.py
```

5. Expose locally for Twilio webhook using ngrok (example):

```
ngrok http 5000
```

6. In Twilio Console, set the WhatsApp sandbox webhook for `WHEN A MESSAGE COMES IN` to:

```
https://<your-ngrok-subdomain>.ngrok.io/text  # for bot_demo.py

or

https://<your-ngrok-subdomain>.ngrok.io/whatsapp  # for research_bot.py
```

## Usage

- Send to your WhatsApp sandbox number:
  - `Explain <topic>` — simple explanation suitable for a 3-year-old
  - `Activities <topic>` — early years activities in two sections, sent as multiple messages if long
  - `tell me more about attention is all you need` — concise scholarly Q&A about the Transformer paper

Research bot examples:

- Search: `transformer attention` or `retrieval augmented generation`
- Select a result: `select 1`
- Start Q&A: `ready for Q&A` or `let's do Q&A`

## Notes

- Responses are chunked to ≤1500 characters to fit WhatsApp limits.
- No multimodal inputs. Tool calling can be added later with Gemini Tool Use if needed.
