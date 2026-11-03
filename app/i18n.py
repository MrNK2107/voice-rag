"""System messages in the asker's language.

Answers themselves are quotes from the user's documents, so they stay in the
document's language unless a translator is configured (app/translate.py).
Everything Vouch says in its own voice (blocked, not in your documents,
emergency advice...) is localised here. Unknown languages fall back to English.

Translations were written for this project and kept deliberately plain; native
review is on the roadmap (docs/PRODUCT.md).
"""
from __future__ import annotations

from typing import Dict, Optional

from app.langid import base_language

# English strings are unchanged from the original pipeline (tests and the
# output-guard templates depend on them).
MESSAGES: Dict[str, Dict[str, str]] = {
    "blocked": {
        "en": "I cannot process that request.",
        "hi": "मैं इस अनुरोध पर काम नहीं कर सकता।",
        "bn": "আমি এই অনুরোধটি প্রক্রিয়া করতে পারি না।",
        "ta": "இந்தக் கோரிக்கையை என்னால் செயல்படுத்த முடியாது.",
        "te": "నేను ఈ అభ్యర్థనను ప్రాసెస్ చేయలేను.",
        "mr": "मी ही विनंती पूर्ण करू शकत नाही.",
        "ur": "میں اس درخواست پر عمل نہیں کر سکتا۔",
        "ar": "لا يمكنني معالجة هذا الطلب.",
        "fa": "نمی‌توانم این درخواست را پردازش کنم.",
        "zh": "我无法处理这个请求。",
        "ja": "このリクエストには対応できません。",
        "ko": "이 요청은 처리할 수 없습니다.",
        "ru": "Я не могу обработать этот запрос.",
        "uk": "Я не можу обробити цей запит.",
        "es": "No puedo procesar esa solicitud.",
        "fr": "Je ne peux pas traiter cette demande.",
        "de": "Diese Anfrage kann ich nicht bearbeiten.",
        "pt": "Não posso processar esse pedido.",
        "it": "Non posso elaborare questa richiesta.",
        "id": "Saya tidak dapat memproses permintaan itu.",
        "tr": "Bu isteği işleyemiyorum.",
        "sw": "Siwezi kushughulikia ombi hilo.",
        "vi": "Tôi không thể xử lý yêu cầu này.",
    },
    "no_context": {
        "en": "I could not find enough relevant context in the {corpus} to answer this.",
        "hi": "{corpus} में इसका उत्तर देने लायक जानकारी नहीं मिली।",
        "bn": "{corpus}-এ এর উত্তর দেওয়ার মতো যথেষ্ট তথ্য পাওয়া যায়নি।",
        "ta": "{corpus} இல் இதற்குப் பதில் சொல்லப் போதுமான தகவல் கிடைக்கவில்லை.",
        "te": "{corpus}లో దీనికి సమాధానం ఇవ్వడానికి సరిపడా సమాచారం దొరకలేదు.",
        "mr": "{corpus} मध्ये याचे उत्तर देण्याइतकी माहिती सापडली नाही.",
        "ur": "{corpus} میں اس کا جواب دینے کے لیے کافی معلومات نہیں ملیں۔",
        "ar": "لم أجد في {corpus} معلومات كافية للإجابة عن هذا السؤال.",
        "fa": "در {corpus} اطلاعات کافی برای پاسخ به این پرسش پیدا نکردم.",
        "zh": "在{corpus}中没有找到足以回答这个问题的内容。",
        "ja": "{corpus}には、この質問に答えられる十分な情報が見つかりませんでした。",
        "ko": "{corpus}에서 이 질문에 답할 만한 내용을 찾지 못했습니다.",
        "ru": "В {corpus} не нашлось достаточно сведений, чтобы ответить на этот вопрос.",
        "uk": "У {corpus} недостатньо відомостей, щоб відповісти на це питання.",
        "es": "No encontré suficiente información en {corpus} para responder a esto.",
        "fr": "Je n'ai pas trouvé assez d'informations dans {corpus} pour répondre.",
        "de": "In {corpus} habe ich nicht genug passende Informationen gefunden, um das zu beantworten.",
        "pt": "Não encontrei informação suficiente em {corpus} para responder a isto.",
        "it": "Non ho trovato abbastanza informazioni in {corpus} per rispondere.",
        "id": "Saya tidak menemukan cukup informasi di {corpus} untuk menjawab ini.",
        "tr": "{corpus} içinde bunu yanıtlamak için yeterli bilgi bulamadım.",
        "sw": "Sikupata taarifa za kutosha katika {corpus} kujibu swali hili.",
        "vi": "Tôi không tìm thấy đủ thông tin trong {corpus} để trả lời câu này.",
    },
    "unanswerable": {
        "en": "The {corpus} has related information, but nothing that answers this exact question.",
        "hi": "{corpus} में इससे जुड़ी जानकारी है, लेकिन ठीक इसी सवाल का जवाब नहीं है।",
        "ta": "{corpus} இல் தொடர்புடைய தகவல் உள்ளது, ஆனால் இந்தக் கேள்விக்கான நேரடிப் பதில் இல்லை.",
        "ur": "{corpus} میں متعلقہ معلومات موجود ہیں، لیکن اس سوال کا براہِ راست جواب نہیں ہے۔",
        "ar": "يحتوي {corpus} على معلومات ذات صلة، لكن لا شيء يجيب عن هذا السؤال تحديدًا.",
        "zh": "{corpus}中有相关信息，但没有直接回答这个问题的内容。",
        "ja": "{corpus}には関連する情報がありますが、この質問への直接の答えはありません。",
        "ko": "{corpus}에 관련 정보는 있지만 이 질문에 대한 직접적인 답은 없습니다.",
        "ru": "В {corpus} есть связанные сведения, но прямого ответа на этот вопрос нет.",
        "es": "{corpus} tiene información relacionada, pero nada que responda exactamente a esta pregunta.",
        "fr": "{corpus} contient des informations proches, mais rien qui réponde exactement à cette question.",
        "de": "{corpus} enthält verwandte Informationen, aber nichts, was genau diese Frage beantwortet.",
        "pt": "{corpus} tem informação relacionada, mas nada que responda exatamente a esta pergunta.",
        "sw": "{corpus} ina taarifa zinazohusiana, lakini hakuna inayojibu swali hili hasa.",
    },
    "ungrounded": {
        "en": "I found related passages, but I cannot produce a sufficiently grounded answer from them.",
        "hi": "मुझे संबंधित अंश मिले, लेकिन उनसे पूरी तरह प्रमाणित उत्तर नहीं बन सका।",
        "ar": "وجدت مقاطع ذات صلة، لكن لا يمكنني تكوين إجابة موثقة بما يكفي منها.",
        "zh": "找到了相关段落，但无法据此给出有充分依据的回答。",
        "ja": "関連する箇所は見つかりましたが、根拠が十分な回答を作れませんでした。",
        "ru": "Я нашёл связанные фрагменты, но не могу составить из них достаточно обоснованный ответ.",
        "es": "Encontré pasajes relacionados, pero no puedo dar una respuesta suficientemente fundamentada con ellos.",
        "fr": "J'ai trouvé des passages proches, mais pas de quoi donner une réponse suffisamment fondée.",
        "de": "Ich habe passende Stellen gefunden, kann daraus aber keine ausreichend belegte Antwort bilden.",
        "pt": "Encontrei trechos relacionados, mas não consigo dar uma resposta suficientemente fundamentada.",
        "ur": "متعلقہ حصے ملے، لیکن ان سے مستند جواب نہیں بن سکا۔",
    },
    "no_audio": {
        "en": "No audio was received. Please record your question again.",
        "hi": "कोई आवाज़ नहीं मिली। कृपया अपना सवाल फिर से रिकॉर्ड करें।",
        "es": "No se recibió audio. Graba tu pregunta de nuevo.",
        "fr": "Aucun son reçu. Enregistrez à nouveau votre question.",
        "ar": "لم يصل أي صوت. يرجى تسجيل سؤالك مرة أخرى.",
    },
    "stt_failed": {
        "en": "Speech-to-text transcription failed.",
    },
    "emergency": {
        "en": "This may be an emergency. Call your local emergency number now (112 in most countries, 911 in North America, 108 in India) or go to the nearest hospital.",
        "hi": "यह आपातकाल हो सकता है। तुरंत आपातकालीन नंबर (भारत में 108 या 112) पर फ़ोन करें या नज़दीकी अस्पताल जाएँ।",
        "bn": "এটি জরুরি অবস্থা হতে পারে। এখনই জরুরি নম্বরে (ভারতে 108 বা 112) ফোন করুন বা নিকটতম হাসপাতালে যান।",
        "ta": "இது அவசர நிலையாக இருக்கலாம். உடனே அவசர எண்ணை (இந்தியாவில் 108 அல்லது 112) அழைக்கவும் அல்லது அருகிலுள்ள மருத்துவமனைக்குச் செல்லவும்.",
        "te": "ఇది అత్యవసర పరిస్థితి కావచ్చు. వెంటనే అత్యవసర నంబర్‌కు (భారత్‌లో 108 లేదా 112) కాల్ చేయండి లేదా దగ్గరలోని ఆసుపత్రికి వెళ్లండి.",
        "mr": "ही आपत्कालीन स्थिती असू शकते. लगेच आपत्कालीन क्रमांकावर (भारतात 108 किंवा 112) फोन करा किंवा जवळच्या रुग्णालयात जा.",
        "ur": "یہ ہنگامی صورتحال ہو سکتی ہے۔ فوراً ایمرجنسی نمبر پر کال کریں یا قریبی ہسپتال جائیں۔",
        "ar": "قد تكون هذه حالة طارئة. اتصل برقم الطوارئ المحلي الآن (112 في معظم الدول، 997 أو 999 في بعض دول الخليج) أو توجّه إلى أقرب مستشفى.",
        "fa": "ممکن است وضعیت اضطراری باشد. همین حالا با شماره اورژانس (115 یا 112) تماس بگیرید یا به نزدیک‌ترین بیمارستان بروید.",
        "zh": "这可能是紧急情况。请立即拨打当地急救电话（中国大陆 120，多数国家 112）或前往最近的医院。",
        "ja": "緊急の可能性があります。すぐに救急（日本では 119）に電話するか、最寄りの病院へ行ってください。",
        "ko": "응급 상황일 수 있습니다. 지금 바로 119에 전화하거나 가까운 병원으로 가세요.",
        "ru": "Это может быть экстренная ситуация. Немедленно позвоните по номеру экстренной помощи (112 или 103) или обратитесь в ближайшую больницу.",
        "uk": "Це може бути надзвичайна ситуація. Негайно телефонуйте за номером екстреної допомоги (112 або 103) або йдіть до найближчої лікарні.",
        "es": "Esto puede ser una emergencia. Llama ahora al número de emergencias local (112 en Europa, 911 en América) o ve al hospital más cercano.",
        "fr": "Il peut s'agir d'une urgence. Appelez immédiatement le numéro d'urgence local (112 en Europe, 15 en France) ou rendez-vous à l'hôpital le plus proche.",
        "de": "Das könnte ein Notfall sein. Rufen Sie sofort den Notruf (112) an oder gehen Sie ins nächste Krankenhaus.",
        "pt": "Isto pode ser uma emergência. Ligue já para o número de emergência local (112 na Europa, 192 no Brasil) ou vá ao hospital mais próximo.",
        "it": "Potrebbe essere un'emergenza. Chiama subito il numero di emergenza (112) o vai all'ospedale più vicino.",
        "id": "Ini mungkin keadaan darurat. Segera hubungi nomor darurat (112 atau 119) atau pergi ke rumah sakit terdekat.",
        "tr": "Bu bir acil durum olabilir. Hemen 112'yi arayın veya en yakın hastaneye gidin.",
        "sw": "Hii inaweza kuwa dharura. Piga simu nambari ya dharura sasa (112 au 999) au nenda hospitali iliyo karibu.",
        "vi": "Đây có thể là trường hợp khẩn cấp. Hãy gọi ngay số cấp cứu (115 hoặc 112) hoặc đến bệnh viện gần nhất.",
    },
    # Self-harm / suicide: shown whatever else happens to the question.
    "crisis": {
        "en": "If you are thinking about suicide or harming yourself, please talk to someone now. Call your local emergency number (112 in most countries, 988 in the US), or in India Tele-MANAS 14416 (free, 24x7).",
        "hi": "अगर आप आत्महत्या या ख़ुद को नुकसान पहुँचाने के बारे में सोच रहे हैं, तो कृपया अभी किसी से बात करें। टेली-मानस 14416 (मुफ़्त, 24x7) या आपातकाल में 108 / 112 पर फ़ोन करें।",
    },
}


def message(key: str, lang: Optional[str] = None, **fmt) -> str:
    """Localised system message; falls back to the base language, then English."""
    table = MESSAGES[key]
    code = lang or "en"
    text = table.get(code) or table.get(base_language(code)) or table["en"]
    if base_language(code) == "hi" and code == "hi-Latn":
        # Romanised Hindi speakers read English more reliably than Devanagari.
        text = table["en"]
    return text.format(**fmt) if fmt else text


def has_message(key: str, lang: Optional[str]) -> bool:
    table = MESSAGES[key]
    return bool(lang) and (lang in table or base_language(lang) in table)
