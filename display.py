"""İngilizce sınıfları Türkçe gösterir; fotoğrafın yanında sonuç paneli oluşturur."""
import html
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

TOOLS = {'grasper': 'Tutucu', 'bipolar': 'Bipolar alet', 'hook': 'Kanca',
         'scissors': 'Makas', 'clipper': 'Klips uygulayıcı', 'irrigator': 'Yıkama / aspirasyon aleti'}
VERBS = {'grasp': 'Tutma', 'retract': 'Çekerek uzaklaştırma', 'dissect': 'Diseksiyon (dokuyu ayırma)',
         'coagulate': 'Koagülasyon', 'clip': 'Klips uygulama', 'cut': 'Kesme',
         'aspirate': 'Aspirasyon (sıvı çekme)', 'irrigate': 'Yıkama', 'pack': 'Yerleştirme',
         'null_verb': 'Eylem belirtilmemiş'}
TARGETS = {'gallbladder': 'Safra kesesi', 'cystic_plate': 'Sistik plak', 'cystic_duct': 'Sistik kanal',
           'cystic_artery': 'Sistik arter', 'cystic_pedicle': 'Sistik pedikül', 'blood_vessel': 'Kan damarı',
           'fluid': 'Sıvı', 'abdominal_wall_cavity': 'Karın duvarı / boşluğu', 'liver': 'Karaciğer',
           'adhesion': 'Yapışıklık', 'omentum': 'Omentum', 'peritoneum': 'Periton', 'gut': 'Bağırsak',
           'specimen_bag': 'Numune torbası', 'null_target': 'Hedef belirtilmemiş'}


def parse_answer(text):
    # HTML boşluğu ve Markdown kod çiti gibi çevre metinlerine tolerans göster.
    text = html.unescape(text).strip()
    start = text.find('{')
    if start < 0:
        raise ValueError('Model yanıtında JSON nesnesi bulunamadı.')
    try:
        result, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as error:
        raise ValueError('Model geçerli veya tamamlanmış JSON üretmedi.') from error
    if not isinstance(result.get('instruments'), list) or not isinstance(result.get('triplets'), list):
        raise ValueError('instruments ve triplets listeleri eksik.')
    if any(not isinstance(x, str) for x in result['instruments']):
        raise ValueError('Alet isimleri metin olmalı.')
    for row in result['triplets']:
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) for k in ('instrument','verb','target')):
            raise ValueError('Üçlü biçimi hatalı.')
    return result


def translate(value, table):
    return table.get(value, f'Bilinmeyen sınıf: {value}')


def turkish_result(result):
    return {
        'aletler': [translate(x, TOOLS) for x in result['instruments']],
        'eylemler': [{'alet': translate(row['instrument'], TOOLS),
                     'eylem': translate(row['verb'], VERBS),
                     'hedef': translate(row['target'], TARGETS)} for row in result['triplets']]
    }


def font(size):
    for path in ['/System/Library/Fonts/Supplemental/Arial.ttf',
                 '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                 'C:/Windows/Fonts/arial.ttf']:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def wrap(text, draw, selected_font, width):
    lines, line = [], ''
    for word in text.split():
        candidate = f'{line} {word}'.strip()
        if line and draw.textlength(candidate, font=selected_font) > width:
            lines.append(line); line = word
        else:
            line = candidate
    return lines + [line]


def save_visual(image_path, result, output_path, adapter_used):
    translated = turkish_result(result)
    with Image.open(image_path) as original:
        picture = original.convert('RGB')
    picture.thumbnail((960, 720))  # Görüntü oranını koru, kırpma yapma.
    normal, title = font(21), font(28)
    panel_width, margin = 500, 28
    measure = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    blocks = [('MODEL TAHMİNİ', title, '#66d9cf'),
              ('Aletler: ' + (', '.join(translated['aletler']) or 'Alet bildirilmedi'), normal, '#ffffff')]
    for i, row in enumerate(translated['eylemler'], 1):
        blocks.extend([(f'{i}. {row["alet"]}', title, '#66d9cf'),
                       ('Eylem: ' + row['eylem'], normal, '#ffffff'),
                       ('Hedef: ' + row['hedef'], normal, '#ffffff')])
    if not translated['eylemler']:
        blocks.append(('Model üçlü bildirmedi.', normal, '#ffffff'))
    blocks.append(('Konum bilgisi yok: numaralar aletlerin görüntüdeki yerini göstermez. Sonuçlar doğrulanmış etiket değildir.', normal, '#aebacf'))
    lines = []
    height = margin
    for text, selected_font, color in blocks:
        for line in wrap(text, measure, selected_font, panel_width - 2*margin):
            lines.append((line, selected_font, color, height))
            height += selected_font.size + 9
        height += 12
    canvas = Image.new('RGB', (picture.width + panel_width, max(picture.height, height + margin)), '#111827')
    canvas.paste(picture, (0, (canvas.height-picture.height)//2))
    draw = ImageDraw.Draw(canvas)
    for text, selected_font, color, y in lines:
        draw.text((picture.width+margin,y), text, font=selected_font, fill=color)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return translated
