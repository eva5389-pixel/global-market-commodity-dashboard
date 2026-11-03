"""Session-local capture of rendered dashboard content for Word exports.

No Streamlit functions are monkey-patched; each script run owns its collector.
"""
from contextvars import ContextVar
from datetime import datetime
from html import unescape
from io import BytesIO
import re
import os
from zoneinfo import ZoneInfo
import pandas as pd
import streamlit as _st

FONT = os.getenv('WORD_REPORT_FONT', 'Noto Sans CJK TC')

_items = ContextVar('word_report_items', default=None)
_settings = ContextVar('word_report_settings', default=None)


def begin_report():
    _items.set([])
    _settings.set([])


def plain(value):
    text = re.sub(r'<(style|script)\b[^>]*>.*?</\1>', '', str(value), flags=re.S|re.I)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\[([^]]+)\]\(([^)]+)\)', r'\1 (\2)', text)
    return unescape(re.sub(r'[*`#]', '', text)).strip()


def record(kind, value):
    if _items.get() is not None:
        _items.get().append((kind, value))


class ReportUI:
    def __init__(self, target, sidebar=False, label=None):
        self.target, self.sidebar_mode, self.label = target, sidebar, label
    def __enter__(self):
        self.target.__enter__()
        if self.label and not self.sidebar_mode: record('heading', self.label)
        return self
    def __exit__(self, *args):
        return self.target.__exit__(*args)
    def __getattr__(self, name):
        target = getattr(self.target, name)
        if name == 'sidebar': return ReportUI(target, True)
        if name in {'columns', 'tabs'}:
            def containers(*args, **kwargs):
                result = target(*args, **kwargs)
                return [ReportUI(c, self.sidebar_mode, args[0][i] if name == 'tabs' else None) for i,c in enumerate(result)]
            return containers
        if name in {'expander', 'container', 'empty'}:
            def container(*args, **kwargs):
                return ReportUI(target(*args, **kwargs), self.sidebar_mode, args[0] if name == 'expander' and args else None)
            return container
        if name in {'selectbox','multiselect','radio','slider','checkbox','text_input','number_input','date_input'}:
            def setting(*args, **kwargs):
                value = target(*args, **kwargs)
                if _settings.get() is not None and args and kwargs.get('type') != 'password':
                    _settings.get().append((plain(args[0]), str(value)))
                return value
            return setting
        if name in {'altair_chart','bar_chart','dataframe','table','title','header','subheader','markdown','caption','write','success','info','warning','error','metric','html'}:
            def display(*args, **kwargs):
                result = target(*args, **kwargs)
                if self.sidebar_mode or not args: return result
                if name == 'altair_chart': record('chart', args[0].to_dict())
                elif name == 'bar_chart':
                    import altair as alt
                    df = pd.DataFrame(args[0]); x=kwargs.get('x'); y=kwargs.get('y')
                    if not x or not y: raise ValueError('Word 匯出需要明確的圖表座標欄位')
                    horizontal=kwargs.get('horizontal',False)
                    enc={'x':alt.X(y+':Q') if horizontal else alt.X(x+':N'), 'y':alt.Y(x+':N') if horizontal else alt.Y(y+':Q')}
                    if kwargs.get('color'):enc['color']=alt.Color(kwargs['color']+':N')
                    record('chart', alt.Chart(df).mark_bar().encode(**enc).properties(height=320).to_dict())
                elif name in {'dataframe','table'}:
                    value=args[0].data if hasattr(args[0], 'data') and not isinstance(args[0],pd.DataFrame) else args[0]
                    record('table', pd.DataFrame(value).copy())
                elif name == 'metric':
                    label=plain(args[0]); value=args[1] if len(args)>1 else kwargs.get('value',''); delta=args[2] if len(args)>2 else kwargs.get('delta')
                    record('text', f'{label}：{value}'+(f'（{delta}）' if delta is not None else ''))
                else:
                    for value in args:
                        if isinstance(value,pd.DataFrame):record('table',value.copy())
                        else:record('heading' if name in {'title','header','subheader'} else 'text',plain(value))
                return result
            return display
        return target


st = ReportUI(_st)


def chart_png(spec):
    import copy
    import vl_convert as vlc
    spec=copy.deepcopy(spec)
    # Use the same data and marks, with a fixed print width and readable fonts.
    spec['width']=900
    spec['background']='white'
    config=spec.setdefault('config',{})
    config['font']=FONT
    config.setdefault('axis',{}).update(labelFontSize=14,titleFontSize=16)
    config.setdefault('legend',{}).update(labelFontSize=14,titleFontSize=16)
    config.setdefault('title',{}).update(fontSize=18)
    return vlc.vegalite_to_png(spec, scale=1.7)


def build_docx(title, items, settings):
    from docx import Document
    from docx.shared import Cm, Pt, RGBColor
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from PIL import Image
    doc=Document(); sec=doc.sections[0]
    sec.page_width=Cm(29.7); sec.page_height=Cm(21)
    sec.top_margin=sec.bottom_margin=Cm(1.5); sec.left_margin=sec.right_margin=Cm(1.7)
    for name in ['Normal','Title','Heading 1','Heading 2']:
        style=doc.styles[name];style.font.name=FONT
        style.font.color.rgb=RGBColor(0,0,0)
        style.font.size=Pt(11 if name=='Normal' else 22 if name=='Title' else 16)
        style.paragraph_format.line_spacing=Pt(17 if name=='Normal' else 26)
        rf=style.element.get_or_add_rPr().get_or_add_rFonts()
        for k in list(rf.attrib):
            if 'Theme' in k:del rf.attrib[k]
        rf.set(qn('w:eastAsia'),FONT)
    for style in doc.styles:
        for border in list(style.element.iter(qn('w:pBdr'))):border.getparent().remove(border)
    doc.add_heading(plain(title)+' 圖表報告',0)
    doc.add_paragraph('匯出時間（台北）：'+datetime.now(ZoneInfo('Asia/Taipei')).strftime('%Y-%m-%d %H:%M:%S'))
    count=sum(kind=='chart' for kind,_ in items)
    doc.add_paragraph(f'包含目前頁面各分頁已產生的 {count} 張圖表與分析。圖表為靜態圖片；資料日期、來源與模型假設依各節標示。未選取的其他市場或功能頁面不在本檔範圍。')
    if settings:
        doc.add_heading('查詢設定',1)
        for label,value in dict(settings).items():doc.add_paragraph(f'{label}：{value}')
    doc.add_page_break()
    chart_count=0
    section_title=plain(title)
    for kind,value in items:
        if kind=='chart':
            png=chart_png(value)  # Fail the entire export if any chart cannot render.
            chart_count+=1
            doc.add_heading(f'圖 {chart_count}　{section_title}',2)
            w,h=Image.open(BytesIO(png)).size
            width=min(26.0,12.7*w/h)
            doc.add_picture(BytesIO(png),width=Cm(width))
            doc.paragraphs[-1].paragraph_format.line_spacing=1.0
            doc.add_page_break()
        elif kind=='heading':
            section_title=plain(value)
            doc.add_heading(section_title,1)
        elif kind=='table' and not value.empty:
            table=doc.add_table(rows=1,cols=len(value.columns));table.style='Table Grid'
            for cell,col in zip(table.rows[0].cells,value.columns):cell.text=str(col)
            table.rows[0]._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
            for row in value.itertuples(index=False,name=None):
                for cell,v in zip(table.add_row().cells,row):cell.text='—' if pd.isna(v) else str(v)
            for row in table.rows:
                for cell in row.cells:
                    for p in cell.paragraphs:
                        p.paragraph_format.line_spacing=Pt(12)
                        for run in p.runs:run.font.size=Pt(8)
        elif kind=='text' and value:
            for line in str(value).splitlines():
                if line.strip():doc.add_paragraph(line.strip())
    # Avoid an empty last page after the final chart.
    last=doc.paragraphs[-1]
    if last._p.xpath('.//w:br[@w:type="page"]'):last._element.getparent().remove(last._element)
    out=BytesIO();doc.save(out);return out.getvalue()


def render_download(title):
    items=list(_items.get() or []); settings=list(_settings.get() or [])
    if not items:return
    count=sum(k=='chart' for k,_ in items)
    _st.divider();_st.subheader('下載 Word 圖表報告')
    _st.caption(f'匯出目前頁面各分頁的 {count} 張圖表、表格與分析；請先完成市場與期間設定。其他功能頁面請切換後分別下載。')
    if _st.button('製作 Word（包含全部圖表）',key='build_word_report'):
        try:
            with _st.spinner('正在將全部圖表嵌入 Word…'):
                data=build_docx(title,items,settings)
            _st.download_button('下載 Word 檔',data=data,file_name=re.sub(r'[^\w\u4e00-\u9fff-]','_',title)+'_'+datetime.now(ZoneInfo('Asia/Taipei')).strftime('%Y%m%d')+'.docx',mime='application/vnd.openxmlformats-officedocument.wordprocessingml.document',on_click='ignore',key='download_word_report')
        except Exception:
            _st.error('Word 製作未完成，為避免漏圖，沒有產生下載檔。請稍後重試或查看主機錯誤紀錄。')
            import logging
            logging.exception('Word report export failed')
