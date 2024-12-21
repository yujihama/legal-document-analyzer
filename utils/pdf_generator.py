import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import plotly.graph_objects as go
import plotly.io as pio
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
        )
        
    def add_title(self, title):
        """タイトルを追加"""
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
        )
        self.elements.append(Paragraph(title, title_style))
        self.elements.append(Spacer(1, 20))
        
    def add_heading(self, text, level=1):
        """見出しを追加"""
        if level == 1:
            style = self.styles['Heading1']
        else:
            style = self.styles['Heading2']
        self.elements.append(Paragraph(text, style))
        self.elements.append(Spacer(1, 12))
        
    def add_paragraph(self, text):
        """段落を追加"""
        self.elements.append(Paragraph(text, self.jp_style))
        self.elements.append(Spacer(1, 12))
        
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
        
    def add_plotly_figure(self, fig, width=500, height=300):
        """Plotlyのグラフを追加"""
        # 一時的な画像ファイルとして保存
        temp_path = f"temp_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        pio.write_image(fig, temp_path, format='png', width=width, height=height)
        
        # 画像をPDFに追加
        img = Image(temp_path, width=width, height=height)
        self.elements.append(img)
        self.elements.append(Spacer(1, 20))
        
        # 一時ファイルを削除
        os.remove(temp_path)
        
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
