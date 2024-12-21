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
    # Noto Fonts
    '/nix/store/*/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    '/nix/store/*/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/nix/store/*/share/fonts/truetype/noto/NotoSansJP-Regular.otf',
    '/nix/store/*/share/fonts/opentype/noto/NotoSansJP-Regular.otf',
    '/nix/store/*/share/fonts/noto-fonts/NotoSans-Regular.ttf',
    '/nix/store/*/share/fonts/noto-fonts-cjk/NotoSansCJK-Regular.ttc',
    # Fallback System Fonts
    '/nix/store/*/share/fonts/truetype/DejaVuSans.ttf',
    '/nix/store/*/share/fonts/truetype/LiberationSans-Regular.ttf',
    # Default Fonts
    '/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
]

def register_fonts():
    """利用可能なフォントを登録して登録されたフォント名のリストを返す"""
    import glob
    registered_fonts = []
    
    logger.info("フォント登録を開始...")
    
    try:
        # システムフォントの検索と登録
        for font_path_pattern in FONT_PATHS:
            logger.debug(f"フォントパスのパターンを検索: {font_path_pattern}")
            
            # グロブパターンを使用してフォントファイルを検索
            found_paths = glob.glob(font_path_pattern)
            if found_paths:
                logger.info(f"見つかったフォントファイル: {found_paths}")
                
                for font_path in found_paths:
                    try:
                        font_name = os.path.splitext(os.path.basename(font_path))[0]
                        if font_name not in registered_fonts:  # 重複を避ける
                            pdfmetrics.registerFont(TTFont(font_name, font_path))
                            registered_fonts.append(font_name)
                            logger.info(f"フォントを登録: {font_name} ({font_path})")
                    except Exception as e:
                        logger.warning(f"フォント登録失敗 ({font_path}): {str(e)}")
                        continue
            else:
                logger.debug(f"フォントが見つかりませんでした: {font_path_pattern}")
        
        if not registered_fonts:
            logger.warning("利用可能なフォントが見つかりませんでした。デフォルトフォントを使用します。")
            try:
                # デフォルトフォントとしてHelveticaを登録
                pdfmetrics.registerFont(TTFont('Helvetica', '/nix/store/*/share/fonts/truetype/DejaVuSans.ttf'))
                registered_fonts.append('Helvetica')
                logger.info("デフォルトフォント(Helvetica)を登録しました")
            except Exception as e:
                logger.error(f"デフォルトフォント登録エラー: {str(e)}")
        
        logger.info(f"登録完了したフォント: {', '.join(registered_fonts)}")
        return registered_fonts
            
    except Exception as e:
        logger.error(f"フォント登録プロセスエラー: {str(e)}", exc_info=True)
        return registered_fonts

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
        """各フォントでPDFレポートを生成"""
        try:
            logger.info("PDFレポートの生成を開始")
            
            # フォントの登録
            registered_fonts = register_fonts()
            if not registered_fonts:
                logger.error("フォントが登録されていません")
                return False

            # 出力ディレクトリの作成
            output_dir = os.path.dirname(self.output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logger.info(f"出力ディレクトリを作成: {output_dir}")

            success = False
            generated_files = []
            
            # 各フォントでPDFを生成
            for font_name in registered_fonts:
                try:
                    # フォント名をファイル名に含める
                    file_name = os.path.splitext(self.output_path)[0]
                    font_pdf_path = f"{file_name}_{font_name}.pdf"
                    
                    logger.info(f"フォント {font_name} でPDFを生成中...")
                    
                    # スタイル設定の更新
                    self.base_style.fontName = font_name
                    self.jp_style.fontName = font_name
                    self.jp_heading_style.fontName = font_name
                    
                    # PDFの生成
                    doc = SimpleDocTemplate(
                        font_pdf_path,
                        pagesize=A4,
                        rightMargin=72,
                        leftMargin=72,
                        topMargin=72,
                        bottomMargin=72
                    )
                    
                    doc.build(self.elements)
                    
                    # 生成結果の確認
                    if os.path.exists(font_pdf_path):
                        size = os.path.getsize(font_pdf_path)
                        logger.info(f"PDF生成完了: {font_pdf_path} ({size:,} bytes)")
                        generated_files.append(font_pdf_path)
                        success = True
                    
                except Exception as e:
                    logger.error(f"フォント {font_name} でのPDF生成エラー: {str(e)}")
                    continue
            
            if success:
                logger.info(f"生成されたPDFファイル: {', '.join(generated_files)}")
                return True
            
            logger.error("PDFファイルを生成できませんでした")
            return False
            
        except Exception as e:
            logger.error(f"PDF生成プロセスエラー: {str(e)}", exc_info=True)
            return False

# フォント登録の実行
if not register_fonts():
    logger.error("フォント登録に失敗しました")