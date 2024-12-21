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
JAPANESE_FONT_NAME = 'NotoSansJP'
JAPANESE_FONT = JAPANESE_FONT_NAME

# 日本語フォントの登録
FONT_PATHS = [
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',  # Noto Sans CJK
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',  # 別のパス
    '/usr/share/fonts/truetype/noto/NotoSansJP-Regular.otf',   # Noto Sans JP
    '/usr/share/fonts/opentype/noto/NotoSansJP-Regular.otf',   # 別のパス
    '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc',        # macOS
    'C:/Windows/Fonts/msgothic.ttc',                          # Windows
    '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc',     # 追加パス
]

def register_japanese_font():
    """利用可能な日本語フォントを検索して登録"""
    global JAPANESE_FONT

    print("日本語フォントの登録を開始...")

    # 利用可能なフォントパスを表示
    for font_path in FONT_PATHS:
        if os.path.exists(font_path):
            print(f"フォントファイルが見つかりました: {font_path}")

    # フォント登録を試行
    for font_path in FONT_PATHS:
        try:
            if os.path.exists(font_path):
                pdfmetrics.registerFont(TTFont(JAPANESE_FONT_NAME, font_path))
                print(f"日本語フォントを登録しました: {font_path}")
                return True
        except Exception as e:
            print(f"フォント登録エラー ({font_path}): {str(e)}")

    print("警告: 日本語フォントが見つかりません。代替フォントを使用します。")
    JAPANESE_FONT = DEFAULT_FONT
    return False

# 利用可能な全てのフォントパスを定義
ALL_FONT_PATHS = {
    'IPAex': [
        '/usr/share/fonts/opentype/ipaexfont/ipaexg.ttf',
        '/usr/share/fonts/opentype/ipaexfont/ipaexm.ttf'
    ],
    'IPA': [
        '/usr/share/fonts/opentype/ipafont/ipag.ttf',
        '/usr/share/fonts/opentype/ipafont/ipam.ttf'
    ],
    'Noto': [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSerifCJK-Regular.ttc'
    ],
    'M+': [
        '/usr/share/fonts/truetype/mplus/mplus-1c-regular.ttf',
        '/usr/share/fonts/truetype/mplus/mplus-2c-regular.ttf'
    ]
}

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

    def try_generate_with_all_fonts(self):
        """全ての利用可能なフォントでPDFの生成を試みる"""
        print("\n=== PDF生成処理を開始します ===")
        
        def debug_font_file(font_path):
            """フォントファイルの詳細情報を表示"""
            if os.path.exists(font_path):
                print(f"フォントファイルが存在します: {font_path}")
                print(f"サイズ: {os.path.getsize(font_path)} bytes")
                print(f"アクセス権限: {oct(os.stat(font_path).st_mode)[-3:]}")
            else:
                print(f"フォントファイルが見つかりません: {font_path}")

        # 各フォントでの生成を試みる
        success_fonts = []
        for font_family, paths in ALL_FONT_PATHS.items():
            print(f"\n--- {font_family}フォントでの生成を試みます ---")
            for font_path in paths:
                try:
                    debug_font_file(font_path)
                    if not os.path.exists(font_path):
                        continue

                    # フォントを登録
                    font_name = f"{font_family}_{os.path.basename(font_path)}"
                    print(f"フォント登録: {font_name}")
                    pdfmetrics.registerFont(TTFont(font_name, font_path))
                    
                    # 各スタイルのフォントを変更
                    original_fonts = {
                        'base': self.base_style.fontName,
                        'jp': self.jp_style.fontName,
                        'heading': self.jp_heading_style.fontName
                    }
                    
                    self.base_style.fontName = font_name
                    self.jp_style.fontName = font_name
                    self.jp_heading_style.fontName = font_name
                    
                    # PDFを生成
                    output_path = f"{self.output_path[:-4]}_{font_family}.pdf"
                    print(f"PDF生成開始: {output_path}")
                    
                    doc = SimpleDocTemplate(
                        output_path,
                        pagesize=A4,
                        rightMargin=72,
                        leftMargin=72,
                        topMargin=72,
                        bottomMargin=72
                    )
                    
                    try:
                        doc.build(self.elements)
                        success_fonts.append((font_family, output_path))
                        print(f"✓ PDF生成成功: {output_path}")
                    except Exception as e:
                        print(f"✗ PDF生成エラー: {str(e)}")
                        continue
                    finally:
                        # スタイルのフォントを元に戻す
                        self.base_style.fontName = original_fonts['base']
                        self.jp_style.fontName = original_fonts['jp']
                        self.jp_heading_style.fontName = original_fonts['heading']
                
                except Exception as e:
                    print(f"✗ フォント処理エラー: {str(e)}")
                    continue
        
        # 結果の表示
        print("\n=== PDF生成結果 ===")
        if success_fonts:
            print("生成に成功したPDFファイル:")
            for font_family, path in success_fonts:
                print(f"- [{font_family}] {path}")
                try:
                    size = os.path.getsize(path)
                    print(f"  サイズ: {size:,} bytes")
                except Exception as e:
                    print(f"  ファイルサイズの取得に失敗: {str(e)}")
        else:
            print("警告: すべてのフォントでPDF生成に失敗しました。")
            print("デフォルトフォントでの生成を試みます...")
            self.generate()

    def generate(self):
        """PDFを生成"""
        try:
            doc = SimpleDocTemplate(
                self.output_path,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            doc.build(self.elements)
            print(f"✓ PDFファイルを生成しました: {self.output_path}")
        except Exception as e:
            print(f"✗ PDF生成エラー: {str(e)}")
            raise

# フォント登録の実行
register_japanese_font()