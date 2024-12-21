import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import datetime

# 日本語フォントの登録（フォールバックメカニズム付き）
def register_japanese_font():
    common_fonts = [
        ('MS-Gothic', 'msgothic.ttc'),
        ('MS-PGothic', 'msgothic.ttc'),
        ('Arial-Unicode-MS', 'arial-unicode-ms.ttf')
    ]
    
    for font_name, font_file in common_fonts:
        try:
            pdfmetrics.registerFont(TTFont(font_name, font_file))
            return font_name
        except:
            continue
    
    # フォールバック：基本フォントを使用
    return 'Helvetica'

default_font = register_japanese_font()

class PDFReportGenerator:
    def __init__(self, output_path):
        self.output_path = output_path
        self.elements = []
        self.styles = getSampleStyleSheet()
        
        # 日本語フォントの設定
        self.jp_style = ParagraphStyle(
            'JapaneseStyle',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=16,
            encoding='utf-8',
            fontName=default_font,  # システムで利用可能な日本語フォント
            wordWrap='CJK',
            allowWidows=1,
            allowOrphans=1,
        )
        
        # 見出しスタイルの設定
        self.heading_style = ParagraphStyle(
            'JapaneseHeading',
            parent=self.styles['Heading1'],
            fontSize=16,
            leading=20,
            encoding='utf-8',
            fontName='Helvetica-Bold',
            wordWrap='CJK',
            spaceAfter=20,
        )
        
    def add_title(self, title):
        """タイトルを追加"""
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            encoding='utf-8',
            fontName='Helvetica-Bold',
            wordWrap='CJK',
            alignment=1,  # 中央揃え
        )
        self.elements.append(Spacer(1, 20))
        self.elements.append(Paragraph(title, title_style))
        self.elements.append(Spacer(1, 30))
        
    def add_heading(self, text, level=1):
        """見出しを追加"""
        style = self.heading_style if level == 1 else ParagraphStyle(
            'JapaneseHeading2',
            parent=self.styles['Heading2'],
            fontSize=14,
            leading=18,
            encoding='utf-8',
            fontName='Helvetica-Bold',
            wordWrap='CJK',
            spaceAfter=15,
        )
        self.elements.append(Spacer(1, 10))
        self.elements.append(Paragraph(text, style))
        self.elements.append(Spacer(1, 15))
        
    def add_paragraph(self, text):
        """段落を追加"""
        self.elements.append(Paragraph(text, self.jp_style))
        self.elements.append(Spacer(1, 10))
        
    def add_table(self, data, col_widths=None):
        """表を追加"""
        if not col_widths:
            col_widths = [inch * 2] * len(data[0])
            
        table = Table(data, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86C1')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), default_font),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#CCCCCC'))
        ]))
        self.elements.append(table)
        self.elements.append(Spacer(1, 20))
        
    def add_pie_chart(self, data, labels, width=400, height=300):
        """円グラフを追加"""
        # グラフ用のスペースを確保
        self.elements.append(Spacer(1, 20))
        
        # グラフの作成
        drawing = Drawing(width, height)
        pie = Pie()
        pie.x = width // 2
        pie.y = height // 2
        pie.width = min(width, height) * 0.5
        pie.height = min(width, height) * 0.5
        pie.data = data
        pie.labels = labels
        pie.slices.strokeWidth = 0.5
        
        # ラベルのスタイル設定
        pie.labels = [f'{label}\n{value}件' for label, value in zip(labels, data)]
        pie.sideLabels = True
        pie.simpleLabels = False
        pie.direction = 'clockwise'
        
        # ポインターラインの設定
        pie.slices.label_pointer_piePad = 10
        pie.slices.label_pointer_edgePad = 25
        
        # 色の設定
        chart_colors = [colors.HexColor('#2E86C1'), colors.HexColor('#E74C3C')]
        for i, color in enumerate(chart_colors):
            pie.slices[i].fillColor = color
        
        drawing.add(pie)
        
        # グラフの追加
        self.elements.append(drawing)
        self.elements.append(Spacer(1, 30))
        
    def generate(self):
        """PDFを生成"""
        doc = SimpleDocTemplate(
            self.output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        doc.build(self.elements)
