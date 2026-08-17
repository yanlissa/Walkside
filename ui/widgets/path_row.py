from PySide6.QtWidgets import (
    QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QVBoxLayout,
)


class PathRow(QFrame):

    def __init__(self, title, placeholder, mode="file"):
        super().__init__()

        self.setObjectName("pathCard")
        self.mode = mode

        self.label = QLabel(title)
        self.label.setObjectName("pathTitle")

        self.input = QLineEdit()
        self.input.setPlaceholderText(placeholder)
        self.input.setFixedHeight(28)

        self.button = QPushButton("…")
        self.button.setObjectName("browseButton")
        self.button.setFixedWidth(36)
        self.button.setFixedHeight(28)
        self.button.clicked.connect(self.choose_path)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(self.input, 1)
        row.addWidget(self.button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 7)
        layout.setSpacing(4)
        layout.addWidget(self.label)
        layout.addLayout(row)

    def choose_path(self):
        if self.mode == "dir":
            path = QFileDialog.getExistingDirectory(self, "Выберите папку")
        elif self.mode == "save":
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Выберите файл результата",
                "",
                "GeoJSON (*.geojson);;JSON (*.json);;Все файлы (*.*)",
            )
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Выберите файл")

        if path:
            self.input.setText(path)

    def text(self):
        return self.input.text().strip()