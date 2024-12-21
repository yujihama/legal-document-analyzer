import os
import logging
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

# ロギングの設定
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# フォント設定
DEFAULT_FONT = 'Helvetica'
JAPANESE_FONT = DEFAULT_FONT

# システムフォントパス
FONT_PATHS = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
    '/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf',
]

def register_fonts():
    """利用可能なフォントを登録"""
    global JAPANESE_FONT
    
    logger.info("フォント登録を開始...")
    
    try:
        # デフォルトフォントの登録
        pdfmetrics.registerFont(TTFont(DEFAULT_FONT, DEFAULT_FONT))
        logger.info(f"デフォルトフォントを登録: {DEFAULT_FONT}")
        
        # システムフォントの検索と登録
        for font_path in FONT_PATHS:
            if os.path.exists(font_path):
                try:
                    font_name = os.path.splitext(os.path.basename(font_path))[0]
                    pdfmetrics.registerFont(TTFont(font_name, font_path))
                    JAPANESE_FONT = font_name
                    logger.info(f"追加フォントを登録: {font_name}")
                    return True
                except Exception as e:
                    logger.warning(f"フォント登録スキップ ({font_path}): {str(e)}")
                    continue
        
        logger.info(f"デフォルトフォントを使用: {DEFAULT_FONT}")
        return True
        
    except Exception as e:
        logger.error(f"フォント登録エラー: {str(e)}")
        return False

# フォントパスはすでに上部で定義済み

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

        # 見出しスタイル（日本語対応）
        self.jp_heading_style = ParagraphStyle(
            'JapaneseHeading',
            parent=self.styles['Heading1'],
            fontSize=16,
            leading=20,
            fontName=JAPANESE_FONT,
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
            fontName=JAPANESE_FONT,
            wordWrap='CJK',
            alignment=1,  # 中央揃え
        )
        self.elements.append(Spacer(1, 20))
        self.elements.append(Paragraph(title, title_style))
        self.elements.append(Spacer(1, 30))

    def add_heading(self, text, level=1):
        """見出しを追加（日本語対応）"""
        style = self.jp_heading_style if level == 1 else ParagraphStyle(
            'Heading2',
            parent=self.styles['Heading2'],
            fontSize=14,
            leading=18,
            fontName=JAPANESE_FONT,
            wordWrap='CJK',
            spaceAfter=15
        )

        try:
            self.elements.append(Spacer(1, 10))
            self.elements.append(Paragraph(text, style))
            self.elements.append(Spacer(1, 15))
        except Exception as e:
            print(f"見出し追加エラー: {str(e)}")
            self.elements.append(Paragraph(text, self.base_style))

    def add_paragraph(self, text):
        """段落を追加（日本語対応）"""
        try:
            p = Paragraph(text, self.jp_style)
            self.elements.append(p)
            self.elements.append(Spacer(1, 10))
        except Exception as e:
            print(f"段落追加エラー: {str(e)}")
            p = Paragraph(text, self.base_style)
            self.elements.append(p)
            self.elements.append(Spacer(1, 10))

    def add_table(self, data, col_widths=None):
        """表を追加"""
        if not col_widths:
            col_widths = [inch * 2] * len(data[0])

        # テーブルスタイルの設定
        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86C1')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), JAPANESE_FONT),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#CCCCCC'))
        ])

        try:
            table = Table(data, colWidths=col_widths, style=table_style)
            self.elements.append(table)
            self.elements.append(Spacer(1, 20))
        except Exception as e:
            print(f"テーブル追加エラー: {str(e)}")

    def add_pie_chart(self, data, labels, width=400, height=300):
        """円グラフを追加"""
        self.elements.append(Spacer(1, 20))

        drawing = Drawing(width, height)
        pie = Pie()
        pie.x = width // 2
        pie.y = height // 2
        pie.width = min(width, height) * 0.5
        pie.height = min(width, height) * 0.5
        pie.data = data
        pie.labels = [f'{label}\n{value}件' for label, value in zip(labels, data)]
        pie.sideLabels = True
        pie.simpleLabels = False

        # 色の設定
        chart_colors = [colors.HexColor('#2E86C1'), colors.HexColor('#E74C3C')]
        for i, color in enumerate(chart_colors):
            pie.slices[i].fillColor = color

        drawing.add(pie)
        self.elements.append(drawing)
        self.elements.append(Spacer(1, 30))

    def generate(self):
        """PDFレポートを生成"""
        try:
            logger.info("PDFレポートの生成を開始")
            
            # 出力ディレクトリの作成
            output_dir = os.path.dirname(self.output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logger.info(f"出力ディレクトリを作成: {output_dir}")
            
            # スタイル設定の確認
            logger.info(f"使用フォント: {JAPANESE_FONT}")
            self.base_style.fontName = JAPANESE_FONT
            self.jp_style.fontName = JAPANESE_FONT
            self.jp_heading_style.fontName = JAPANESE_FONT
            
            # PDFの生成
            doc = SimpleDocTemplate(
                self.output_path,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            doc.build(self.elements)
            
            # 生成結果の確認
            if os.path.exists(self.output_path):
                size = os.path.getsize(self.output_path)
                logger.info(f"PDF生成完了: {self.output_path} ({size:,} bytes)")
                return True
                
            logger.error("PDFファイルが生成されませんでした")
            return False
            
        except Exception as e:
            logger.error(f"PDF生成エラー: {str(e)}", exc_info=True)
            return False

# フォント登録の実行
if not register_fonts():
    logger.error("フォント登録に失敗しました")