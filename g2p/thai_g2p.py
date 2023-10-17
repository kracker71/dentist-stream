import os
import subprocess

from optparse import Values
import SequiturTool
from sequitur import Translator

class ThaiG2P(object):
    def __init__(self):
        self._file_path = os.path.dirname(os.path.abspath(__file__))
        self._model_path = os.path.join(self._file_path, 'model/thai-g2p-model')
        self._dict_path = os.path.join(self._file_path, 'dict/dic_g2p_th.txt')
        
        sequitur_options = Values()
        sequitur_options.resume_from_checkpoint = False
        sequitur_options.modelFile = self._model_path
        sequitur_options.shouldRampUp = False
        sequitur_options.trainSample = False
        sequitur_options.shouldTranspose = False
        sequitur_options.newModelFile = False
        sequitur_options.shouldSelfTest = False
        model_g2p = SequiturTool.procureModel(sequitur_options, None)
        self.translator= Translator(model_g2p)
    
        # Load g2p dict
        self.g2p_dict = dict()
        with open(self._dict_path, "r", encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                s = line.strip().split(' ')
                word, pho = s[0], s[1:]
                self.g2p_dict[word] = pho
                
    def __call__(self, word):
        """
        return phonemes of word
        if word not found in g2p common dict then the model will predict the phonemes.
        """
        phonemes = list()
        if word in self.g2p_dict.keys():
            phonemes = self.g2p_dict[word]
        else:
            # cmd = "g2p.py --encoding utf-8 --model {} --word {}".format(self._model_path, word)
            # direct_output = subprocess.check_output(cmd, shell=True)
            
            # _text = direct_output.decode('utf-8')
            phonemes = list(self.translator(word))
            # word, pho = _text.strip().split('\t')
        
            with open(self._dict_path, 'a') as f:
                f.write(word+' '+' '.join(phonemes)+'\n')
                
            self.g2p_dict[word] = phonemes
            # phonemes = pho.split(' ')
            
        return phonemes

if __name__ == "__main__":
    import time
    
    texts = [
        'แมวสะเตอร์'
    #     "มี พระบรมราชโองการ โปรดเกล้า โปรดกระหม่อม พระราชทาน ยศ ทหาร",
    # "ชำแหละ ชัด แมนยู หัก เขา แกะ ฉลุย เอฟ เอคัพ",
    # "ศึก แมนเชสเตอร์ ดาร์บี้ ใน ฟุตบอล พรีเมียร์ลีก อังกฤษ",
    # "ยูทูบเบอร์ แดน ปลาดิบ คิด อุตริ ใช้ ฮาร์ดแวร์ ที่ ความร้อน สูง ปรุง สุก อาหาร",
    # "มิสยูนิเวิร์ส ขาย คฤหาสน์ สุด หรู เหตุ ใกล้ วัด ไทย ตี ระฆัง ดัง ทุก เช่า",
    # "กษัตริย์ นอก ขัตติยะ และ คนมหากาฬ ถอน ราก ทรชน ฉาย แล้ว วัน พรุ่งนี้",
    # "นิโคติน จะ ออกฤทธิ์ ผ่าน ทาง ตัว รับ แอเซทิลโคลีน แบบ นิโคตินิก",
    # "ศิษย์ เชื่อ ฤาษี พระเกจิ อาจารย์ดัง ที่ อินเดีย ยัง ไม่ ดับ แต่ บำเพ็ญ ตบะ ลึก",
    # "ผลงาน ชิ้น เอก ของ พระสุนทรโวหาร กวีเอก แห่ง กรุงรัตนโกสินทร์",
    # "ที่ โทร มา หา ก็ ไม่ มี ไร มาก หรอก ถ้า จะ ให้ มี ก็ มี แต่ คิด ถุง",
    # "รอง ปลัด ดิจิทัล เผย คุม ตัว เยาวชน ร่วม กด เอฟ ห้า ปรับ ทัศนคติ แล้ว",
    # "จรรยาบรรณ หมายถึง ประมวล กฎเกณฑ์ ความประพฤติ",
    # "ไก่ งาม เพราะ ขน คน จน เพราะ กรรม ขี่ จรวด",
    # "ขอ ความกรุณา อย่า ดูด บุหรี่ และ ซิก้าร์ ใน บริเวณ นี่ นะ ค่ะ",
    # "วัน นี้ เหมือน น้ำ รด ลง บน ต้นไม้ แห้ง เหี่ยว ใน ใจ พวก นก",
    # "โคตร สุด พลุ แตก ปัง ไม่ หยุด ยิ้ม แก้ม แตก ยิ้ม หน้า บาน",
    # "พรุ่ง นี้ มี ฤกษ์ ดี เรา จะ ขี่ เรือ บรรทุก ไป ซื้อ โรตี สด ที่ เพชรบูรณ์",
    # "ฆราวาส คือ ผู้ อยู่ ครอง เรือน มิ ใช่ บรรพชิต",
    # "สมัคร แอคเคานต์ ผ่าน ระบบ แอนดรอยด์ ได้ รับ สิทธ์ เอกซเรย์ มะเร็ง ไส้ติ่ง ฟรี",
    # "บัตร เครดิต เดียว ที่ ตอบ โจทย์ ทุก ไลฟ์สไตล์ เลือก ได้ ใน แบบ คุณ",
    # "มี เงินเดือน เริ่มต้น หนึ่ง หมื่น ห้า พัน บาท ก็ สมัคร บัตร เครดิต ได้ แล้ว นะ",
    # "หาก ไม่ สามารถ ใช้ งาน แอปพลิเคชัน ได้ โปรด ติดต่อ มา ทาง เฟซบุ๊ก ไลน์ หรือ อีเมล",
    # "สงสัย วันนี้ จะ มี คน นก โดน ผู้ เทมา แน่นอน สะใจ สิบ กระโหลก",
    # "ทำไม เรา มา นั่ง เม้าท์ มอย เรื่อง เจ้า คน สอง ซิม ยืน หนึ่ง เรื่อง หัว ร้อน",
    # "ศาสตราจารย์ และ อาชญากร อีก แปด คน ร่วม ปฏิบัติการ โจรกรรม ซึ่ง มุ่ง เป้า ไป ที่ โรงกษาปณ์ ของ สเปน",
    # 'เขา นั่ง ตา กลม อยู่ ข้าง ป้าย กลับ รถ',
]

    g2p_th = ThaiG2P()
            
    # beg_t = time.time()
    
    for text in texts:
        print(g2p_th(text))
    # word_phn = list()
    
    # with open('test.txt', 'w') as txt:
    #     for text in texts:
    #         _phn = list()
    #         for word in text.split(' '):
    #             phn = g2p_th(word)
    #             _phn+=phn
    #         txt.write(' '.join(_phn)+'\n')
    #     # word_phn.append(_phn)
        
    #     # print(phn)
    # print(time.time()-beg_t)
