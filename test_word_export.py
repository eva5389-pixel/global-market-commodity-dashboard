import io
import unittest
from zipfile import ZipFile
from unittest.mock import patch
import altair as alt
import pandas as pd
import word_export as w
class WordTests(unittest.TestCase):
 def test_all_images_embedded(self):
  c=alt.Chart(pd.DataFrame({'日期':['A','B'],'價格':[2,3]})).mark_bar().encode(x='日期:N',y='價格:Q').to_dict()
  b=w.build_docx('測試',[('heading','技術分析'),('chart',c),('chart',c),('text','來源：測試資料')],[('市場','台灣')])
  with ZipFile(io.BytesIO(b)) as z:
   xml=z.read('word/document.xml').decode();self.assertEqual(xml.count('<wp:inline'),2);self.assertIn('來源：測試資料',xml)
 def test_no_partial_export(self):
  with patch.object(w,'chart_png',side_effect=ValueError('bad chart')):
   with self.assertRaises(ValueError):w.build_docx('測試',[('chart',{})],[])
 def test_reset_and_style_removal(self):
  w.begin_report();w.record('text','A');w.begin_report();self.assertEqual(w._items.get(),[])
  self.assertEqual(w.plain('<style>.x{color:red}</style><div>市場</div>'),'市場')
if __name__=='__main__':unittest.main()
