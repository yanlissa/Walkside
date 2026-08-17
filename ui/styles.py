STYLE = """
QMainWindow {
    background: #0b111b;
}

QWidget#central {
    background: #0b111b;
}

QWidget {
    color: #e5e7eb;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 10.5pt;
}

QLabel {
    background: transparent;
    border: none;
}

QFrame#topTitle {
    background: #0b111b;
}

QFrame#body {
    background: #0b111b;
}

QSplitter#bodySplitter {
    background: #0b111b;
}

QSplitter#bodySplitter::handle {
    background: #0b111b;
}

QSplitter#bodySplitter::handle:hover {
    background: #172235;
}

QFrame#files_path,
QFrame#progress {
    background: #111827;
    border: 1px solid #243244;
    border-radius: 12px;
}

QFrame#taskbar,
QFrame#logs {
    background: transparent;
    border: none;
}

QScrollArea#leftScroll {
    background: transparent;
    border: none;
}

QWidget#leftScrollViewport {
    background: transparent;
}

QWidget#leftScrollInner {
    background: transparent;
}

QLabel#title {
    color: #ffffff;
    font-size: 26px;
    font-weight: 800;
}

QLabel#stateTitle {
    color: #ffffff;
    font-size: 22px;
    font-weight: 700;
}

QLabel#sectionTitle {
    color: #ffffff;
    font-size: 15px;
    font-weight: 700;
}

QLabel#muted {
    color: #9ca3af;
}

QLabel#pathTitle {
    color: #f3f4f6;
    font-weight: 600;
}

QLabel#statTitle {
    color: #f3f4f6;
}

QLabel#statValue {
    color: #e5e7eb;
}

QFrame#pathCard {
    background: #151e2d;
    border: 1px solid #2f4058;
    border-radius: 9px;
}

QLineEdit {
    background: #09111d;
    color: #e5e7eb;
    border: 1px solid #334155;
    border-radius: 7px;
    padding-left: 10px;
    padding-right: 10px;
    selection-background-color: #5b8cff;
}

QLineEdit:hover {
    border: 1px solid #465a78;
}

QLineEdit:focus {
    background: #0b1422;
    border: 1px solid #5b8cff;
}

QPushButton {
    background: #24324a;
    color: #ffffff;
    border: 1px solid #334766;
    border-radius: 9px;
    padding: 7px 10px;
    font-weight: 600;
}

QPushButton:hover {
    background: #2f4262;
}

QPushButton:pressed {
    background: #1f2d43;
}

QPushButton:disabled {
    background: #1b2433;
    color: #6b7280;
    border: 1px solid #253044;
}

QPushButton#browseButton {
    font-size: 18px;
    padding: 0px;
}

QPushButton#primary {
    background: #5b8cff;
    color: #06101f;
    border: 1px solid #77a0ff;
    font-weight: 800;
}

QPushButton#primary:hover {
    background: #72a0ff;
}

QPushButton#danger {
    background: #673341;
    border: 1px solid #7c4051;
}

QPushButton#danger:hover {
    background: #7b3b4d;
}

QProgressBar {
    background: #0b111b;
    color: #ffffff;
    border: 1px solid #334155;
    border-radius: 8px;
    text-align: center;
    font-weight: 600;
}

QProgressBar::chunk {
    background: #5b8cff;
    border-radius: 7px;
}

QTextEdit {
    background: #0b111b;
    color: #dbeafe;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 8px;
    font-family: "Consolas", "Cascadia Mono", monospace;
    font-size: 12px;
}

QScrollBar:vertical {
    background: transparent;
    width: 8px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #334155;
    min-height: 24px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background: #465a78;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {
    background: transparent;
}

QLineEdit:disabled {
    background: #111827;
    color: #6b7280;
    border: 1px solid #253044;
}

QFrame#pathCard:disabled {
    background: #111827;
    border: 1px solid #253044;
}

"""