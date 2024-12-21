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

class PDFReportGenerator:
    def __init__(self, output_path):
        self.output_path = output_path
        self.elements = []
        self.styles = getSampleStyleSheet()
        
        # 日本語フォントのスタイル設定
        self.jp_style = ParagraphStyle(
            'JapaneseStyle',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=14,
            encoding='utf-8',
            fontName='Helvetica',
            wordWrap='CJK',
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
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
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
        pie.y = height // 2 - 20  # 少し上に調整
        pie.width = min(width, height) * 0.6
        pie.height = min(width, height) * 0.6
        pie.data = data
        pie.labels = labels
        pie.slices.strokeWidth = 0.5
        
        # ラベルのスタイル設定
        pie.labels = [f'{label}\n{value}件' for label, value in zip(labels, data)]
        pie.labelRadius = 1.2
        pie.fontSize = 10
        
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
