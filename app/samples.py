"""Example collection: "Lakeview Civic Help Desk".

A fictional city's help-desk handbook, one short passage per language, written
for this repository (Lakeview is not a real city; the phone numbers use the
reserved 555 range). It shows the product's core promise in one screen: ask in
your language, get a quote from the passage written in it, with the source.
"""
from __future__ import annotations

from typing import Dict, List

from app.chunking import RawDoc

CITY_COLLECTION = "city-desk"
CITY_TITLE = "Lakeview Civic Help Desk"
CITY_DESCRIPTION = "Sample: a fictional city's help-desk handbook, one passage per language."


def _d(doc_id: str, title: str, lang: str, text: str) -> RawDoc:
    return RawDoc(doc_id, text, title=title, language=lang, source_type="sample",
                  metadata={"source": f"Lakeview handbook / {title}"})


CITY_DOCS: List[RawDoc] = [
    _d("waste_en", "Waste collection", "en",
       "Household waste is collected every Tuesday and Friday morning. Put your bins out before 7 am on collection days. "
       "Recycling such as paper, glass and cans goes in the blue bin, which is collected every second Wednesday."),
    _d("library_es", "Biblioteca Central", "es",
       "La Biblioteca Central de Lakeview abre de lunes a sábado, de 9 a 20 horas, y los domingos permanece cerrada. "
       "El carné de biblioteca es gratuito para todos los residentes y se obtiene presentando un documento de identidad."),
    _d("birth_fr", "Acte de naissance", "fr",
       "Pour obtenir un acte de naissance, présentez-vous au guichet de l'état civil de la mairie avec une pièce d'identité. "
       "L'acte de naissance est délivré gratuitement le jour même. Vous pouvez aussi faire la demande en ligne sur le portail de la ville."),
    _d("water_ar", "فاتورة المياه", "ar",
       "يمكن دفع فاتورة المياه عبر الإنترنت أو في مكتب خدمة العملاء في مبنى البلدية. تصدر فواتير المياه كل شهرين. "
       "في حال وجود تسرب للمياه اتصل بخط الطوارئ على الرقم 555-0100 على مدار الساعة."),
    _d("bus_zh", "公交月票", "zh",
       "学生和六十五岁以上的老人可以申请半价公交月票。申请时请携带身份证件和一张照片到市交通服务中心办理。月票在全市所有公交线路上有效。"),
    _d("tax_hi", "संपत्ति कर", "hi",
       "संपत्ति कर साल में दो बार, अप्रैल और अक्टूबर में जमा करना होता है। 30 अप्रैल से पहले पूरा संपत्ति कर जमा करने पर 5 प्रतिशत की छूट मिलती है। "
       "संपत्ति कर नगर निगम के पोर्टल पर ऑनलाइन जमा किया जा सकता है।"),
    _d("snow_ru", "Уборка снега", "ru",
       "Зимой городские службы очищают тротуары от снега до 8 утра. Владельцы домов обязаны расчистить вход в свой дом "
       "в течение 24 часов после снегопада. Сообщить о неубранной улице можно по телефону 311."),
    _d("clinic_sw", "Kliniki ya afya", "sw",
       "Kliniki ya afya ya jamii iko wazi kila siku kuanzia saa mbili asubuhi hadi saa kumi jioni. "
       "Chanjo za watoto hutolewa bure kila Jumatatu na Alhamisi. Tafadhali leta kadi ya chanjo ya mtoto."),
    _d("quake_ja", "地震のとき", "ja",
       "地震が起きたら、まず机の下に隠れて頭を守ってください。揺れが収まったら、最寄りの避難所である中央公園に移動してください。避難所には水と毛布が用意されています。"),
    _d("business_pt", "Licença comercial", "pt",
       "Para abrir um pequeno negócio é preciso registar a empresa no balcão de licenças da câmara municipal. "
       "A licença comercial custa 50 dólares por ano e é emitida em até cinco dias úteis."),
    _d("voter_ur", "ووٹر رجسٹریشن", "ur",
       "ووٹر رجسٹریشن کے لیے شہری مرکز میں فارم جمع کرائیں۔ درخواست کے ساتھ شناختی کارڈ اور رہائش کا ثبوت لگانا ضروری ہے۔ ووٹر رجسٹریشن مفت ہے۔"),
    _d("quiet_de", "Ruhezeiten", "de",
       "Die Ruhezeiten in Lakeview gelten werktags von 22 bis 7 Uhr und am Sonntag ganztägig. "
       "Lärmbeschwerden können rund um die Uhr unter der Nummer 311 gemeldet werden."),
    _d("bulky_ko", "대형 폐기물", "ko",
       "대형 폐기물은 온라인으로 수거를 신청해야 합니다. 대형 폐기물 수거 비용은 품목당 10달러이며 신청 후 3일 이내에 수거됩니다."),
    _d("story_ta", "கதை நேரம்", "ta",
       "நகர நூலகத்தில் குழந்தைகளுக்கான கதை நேரம் ஒவ்வொரு சனிக்கிழமையும் காலை 10 மணிக்கு நடைபெறும். கதை நேரம் இலவசம், முன்பதிவு தேவையில்லை."),
]

# One question per language for the UI's suggestion chips and the tests.
CITY_QUESTIONS: List[Dict[str, str]] = [
    {"query": "When is household waste collected?", "lang": "en", "doc": "waste_en"},
    {"query": "¿A qué hora abre la biblioteca central?", "lang": "es", "doc": "library_es"},
    {"query": "Comment obtenir un acte de naissance ?", "lang": "fr", "doc": "birth_fr"},
    {"query": "كيف أدفع فاتورة المياه؟", "lang": "ar", "doc": "water_ar"},
    {"query": "老人怎么办理半价公交月票？", "lang": "zh", "doc": "bus_zh"},
    {"query": "संपत्ति कर पर छूट कब मिलती है?", "lang": "hi", "doc": "tax_hi"},
    {"query": "Как сообщить о неубранной улице?", "lang": "ru", "doc": "snow_ru"},
    {"query": "Kliniki ya afya iko wazi saa ngapi?", "lang": "sw", "doc": "clinic_sw"},
    {"query": "ووٹر رجسٹریشن کے لیے کیا چاہیے؟", "lang": "ur", "doc": "voter_ur"},
]
