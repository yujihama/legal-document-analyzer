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

# フォント定数とスタイル設定
DEFAULT_FONT = 'Helvetica'
JAPANESE_FONT = 'HeiseiKakuGo-W5'

# 日本語フォントの登録
try:
    pdfmetrics.registerFont(TTFont('HeiseiKakuGo-W5', 'HeiseiKakuGo-W5'))
    print("日本語フォントが正常に登録されました")
except Exception as e:
    print(f"日本語フォント登録エラー: {str(e)}")
    # フォールバック: 組み込みのHelveticaフォントを使用
    JAPANESE_FONT = 'Helvetica'

class PDFReportGenerator:
    def __init__(self, output_path):
        self.output_path = output_path
        self.elements = []
        self.styles = getSampleStyleSheet()
        
        # 基本スタイル（英数字用）
        self.base_style = ParagraphStyle(
            'BaseStyle',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=16,
            fontName=DEFAULT_FONT
        )
        
        # 日本語用スタイル
        self.jp_style = ParagraphStyle(
            'JapaneseStyle',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=16,
            fontName=JAPANESE_FONT,
            wordWrap='CJK',
            allowWidows=1,
            allowOrphans=1,
            spaceAfter=10,
            spaceBefore=10,
            encoding='utf-8'
        )
        
        # 見出しスタイルの設定（日本語対応）
        self.heading_style = ParagraphStyle(
            'BaseHeading',
            parent=self.styles['Heading1'],
            fontSize=16,
            leading=20,
            fontName='Helvetica-Bold',
            spaceAfter=20
        )
        
        # 日本語見出しスタイル
        self.jp_heading_style = ParagraphStyle(
            'JapaneseHeading',
            parent=self.styles['Heading1'],
            fontSize=16,
            leading=20,
            fontName=DEFAULT_FONT,
            wordWrap='CJK',
            spaceAfter=20
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
        """見出しを追加（日本語対応）"""
        # 日本語テキストかどうかをチェック
        try:
            text.encode('ascii')
            is_japanese = False
        except UnicodeEncodeError:
            is_japanese = True
        
        if level == 1:
            style = self.jp_heading_style if is_japanese else self.heading_style
        else:
            style = ParagraphStyle(
                'Heading2',
                parent=self.styles['Heading2'],
                fontSize=14,
                leading=18,
                fontName=DEFAULT_FONT if is_japanese else 'Helvetica-Bold',
                wordWrap='CJK' if is_japanese else None,
                spaceAfter=15
            )
        
        try:
            self.elements.append(Spacer(1, 10))
            self.elements.append(Paragraph(text, style))
            self.elements.append(Spacer(1, 15))
        except Exception as e:
            # エラーが発生した場合、基本スタイルでフォールバック
            fallback_style = self.heading_style
            fallback_text = text.encode('utf-8', errors='ignore').decode('utf-8')
            self.elements.append(Spacer(1, 10))
            self.elements.append(Paragraph(fallback_text, fallback_style))
            self.elements.append(Spacer(1, 15))
        
    def add_paragraph(self, text):
        """段落を追加（日本語対応）"""
        # スタイルの選択（日本語文字が含まれているかどうかで判断）
        style = self.base_style
        try:
            # テキストをエンコード/デコードしてUnicodeチェック
            text.encode('ascii')
        except UnicodeEncodeError:
            # 日本語が含まれる場合、日本語用スタイルを使用
            style = self.jp_style
        
        # 段落の追加
        try:
            p = Paragraph(text, style)
            self.elements.append(p)
            self.elements.append(Spacer(1, 10))
        except Exception as e:
            # エラーが発生した場合、代替テキストを使用
            fallback_text = text.encode('utf-8', errors='ignore').decode('utf-8')
            p = Paragraph(fallback_text, self.base_style)
            self.elements.append(p)
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
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
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
