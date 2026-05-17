from __future__ import annotations

import logging
import threading
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from beam_optimizer.integrations import KompasAdapter
from beam_optimizer.materials import get_material_by_title, material_titles
from beam_optimizer.models import (
    BeamRequest,
    Constraints,
    LoadCase,
    LoadType,
    NumericRange,
    OptimizationGoal,
    SectionSearchSpace,
    SectionType,
    SupportType,
)
from beam_optimizer.optimizer import run_optimization
from beam_optimizer.reporting import clear_output_structure, configure_logging, ensure_output_structure, save_run_report

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path.cwd() / "output"
DEFAULT_RANGE_SUBDIVISIONS = 4

SECTION_OPTIONS = {
    "Прямоугольное": SectionType.RECTANGULAR,
    "Квадратное": SectionType.SQUARE,
    "Круглое сплошное": SectionType.SOLID_ROUND,
    "Круглая труба": SectionType.ROUND_TUBE,
}

SUPPORT_OPTIONS = {
    "Балка на двух опорах": SupportType.SIMPLY_SUPPORTED,
    "Консоль": SupportType.CANTILEVER,
    "Жесткое защемление с двух сторон": SupportType.FIXED_FIXED,
}

LOAD_OPTIONS = {
    "Сосредоточенная сила в центре": LoadType.CENTER_POINT,
    "Сосредоточенная сила в точке": LoadType.POINT_AT_POSITION,
    "Равномерно распределенная нагрузка": LoadType.UNIFORM_DISTRIBUTED,
}

GOAL_OPTIONS = {
    "Самая лёгкая балка": OptimizationGoal.MIN_WEIGHT,
    "Самый большой запас прочности": OptimizationGoal.MAX_SAFETY_FACTOR,
}

GOAL_DESCRIPTIONS = {
    "Самая лёгкая балка": "Сортировка по минимальной массе. Ограничения всё равно проверяются, но если ни один вариант их не проходит, будут показаны самые лёгкие.",
    "Самый большой запас прочности": "Сортировка по максимальному коэффициенту запаса. Полезно, когда масса вторична.",
}

SECTION_DIMENSION_LABELS = {
    SectionType.RECTANGULAR: ("Мин. ширина, мм", "Макс. ширина, мм", "Мин. высота, мм", "Макс. высота, мм"),
    SectionType.SQUARE: ("Мин. сторона, мм", "Макс. сторона, мм", "", ""),
    SectionType.SOLID_ROUND: ("Мин. диаметр, мм", "Макс. диаметр, мм", "", ""),
    SectionType.ROUND_TUBE: (
        "Мин. внешний диаметр, мм",
        "Макс. внешний диаметр, мм",
        "Мин. толщина стенки, мм",
        "Макс. толщина стенки, мм",
    ),
}

SINGLE_DIMENSION_SECTIONS = {SectionType.SQUARE, SectionType.SOLID_ROUND}


class BeamOptimizerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Подбор балки и создание модели в КОМПАС")
        self.root.geometry("1320x860")

        self.output_dir = DEFAULT_OUTPUT_DIR
        self.length_var = tk.StringVar(value="2.0")
        self.section_var = tk.StringVar(value="Прямоугольное")
        self.material_var = tk.StringVar(value="Сталь")
        self.support_var = tk.StringVar(value="Балка на двух опорах")
        self.load_var = tk.StringVar(value="Сосредоточенная сила в центре")
        self.load_magnitude_var = tk.StringVar(value="10000")
        self.load_position_var = tk.StringVar(value="1.0")
        self.primary_min_var = tk.StringVar(value="40")
        self.primary_max_var = tk.StringVar(value="200")
        self.secondary_min_var = tk.StringVar(value="60")
        self.secondary_max_var = tk.StringVar(value="300")
        self.max_stress_var = tk.StringVar(value="160")
        self.max_deflection_var = tk.StringVar(value="5")
        self.min_safety_var = tk.StringVar(value="1.5")
        self.max_primary_dim_var = tk.StringVar(value="")
        self.max_secondary_dim_var = tk.StringVar(value="")
        self.goal_var = tk.StringVar(value="Самая лёгкая балка")
        self.goal_description_var = tk.StringVar(value=GOAL_DESCRIPTIONS[self.goal_var.get()])
        self.top_n_var = tk.StringVar(value="5")
        self.create_model_var = tk.BooleanVar(value=True)

        self.primary_label_var = tk.StringVar(value="Мин. ширина, мм")
        self.primary_max_label_var = tk.StringVar(value="Макс. ширина, мм")
        self.secondary_label_var = tk.StringVar(value="Мин. высота, мм")
        self.secondary_max_label_var = tk.StringVar(value="Макс. высота, мм")
        self.load_magnitude_label_var = tk.StringVar(value="Сила, Н")

        self.current_candidates = []
        self.current_report_path: Path | None = None
        self.cad_adapter: KompasAdapter | None = None

        self._optimization_running = False

        self._build_layout()
        self._bind_events()
        self._refresh_dynamic_fields()

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        top_frame = ttk.Frame(container)
        top_frame.pack(fill=tk.X)
        bottom_frame = ttk.Frame(container)
        bottom_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        self._build_form(top_frame)
        self._build_results(bottom_frame)

    def _build_form(self, parent: ttk.Frame) -> None:
        columns = [ttk.Frame(parent) for _ in range(4)]
        for index, column in enumerate(columns):
            column.grid(row=0, column=index, sticky="nsew", padx=(0 if index == 0 else 8, 0))
            parent.columnconfigure(index, weight=1)

        self._build_general_frame(columns[0])
        self._build_search_space_frame(columns[1])
        self._build_constraints_frame(columns[2])
        self._build_actions_frame(columns[3])

    def _build_general_frame(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Исходные данные", padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        self._add_labeled_entry(frame, "Длина балки, м", self.length_var, row=0)
        self._add_labeled_combo(frame, "Тип сечения", self.section_var, list(SECTION_OPTIONS.keys()), row=1)
        self._add_labeled_combo(frame, "Материал", self.material_var, material_titles(), row=2)
        self._add_labeled_combo(frame, "Схема закрепления", self.support_var, list(SUPPORT_OPTIONS.keys()), row=3)
        self._add_labeled_combo(frame, "Тип нагрузки", self.load_var, list(LOAD_OPTIONS.keys()), row=4)
        self._add_labeled_entry(frame, self.load_magnitude_label_var, self.load_magnitude_var, row=5)
        self.position_entry = self._add_labeled_entry(frame, "Координата нагрузки, м", self.load_position_var, row=6)

    def _build_search_space_frame(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Перебор размеров сечения", padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        self._add_labeled_entry(frame, self.primary_label_var, self.primary_min_var, row=0)
        self._add_labeled_entry(frame, self.primary_max_label_var, self.primary_max_var, row=1)
        self.secondary_row_widgets = [
            self._add_labeled_entry_widgets(frame, self.secondary_label_var, self.secondary_min_var, row=3),
            self._add_labeled_entry_widgets(frame, self.secondary_max_label_var, self.secondary_max_var, row=4),
        ]

    def _build_constraints_frame(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Ограничения и критерий выбора", padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        self._add_labeled_entry(frame, "Макс. напряжение, МПа (пусто = без лим.)", self.max_stress_var, row=0)
        self._add_labeled_entry(frame, "Макс. прогиб, мм (пусто = без лим.)", self.max_deflection_var, row=1)
        self._add_labeled_entry(frame, "Мин. коэф. запаса (пусто = без лим.)", self.min_safety_var, row=2)
        self._add_labeled_entry(frame, "Макс. осн. размер, мм (пусто = без лим.)", self.max_primary_dim_var, row=3)
        self._add_labeled_entry(frame, "Макс. доп. размер, мм (пусто = без лим.)", self.max_secondary_dim_var, row=4)
        self._add_labeled_combo(frame, "Как выбрать лучший вариант", self.goal_var, list(GOAL_OPTIONS.keys()), row=5)
        description_label = ttk.Label(
            frame,
            textvariable=self.goal_description_var,
            wraplength=280,
            justify=tk.LEFT,
        )
        description_label.grid(row=12, column=0, sticky="w", pady=(2, 8))
        self._add_labeled_entry(frame, "Сколько лучших вариантов показать", self.top_n_var, row=6)

    def _build_actions_frame(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Запуск", padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text=f"Результаты всегда сохраняются в: {self.output_dir}", wraplength=280, justify=tk.LEFT).grid(
            row=0, column=0, sticky="w", pady=(0, 10)
        )
        ttk.Checkbutton(frame, text="Создавать модель в КОМПАС", variable=self.create_model_var).grid(
            row=1, column=0, sticky="w", pady=(0, 10)
        )
        self._run_button = ttk.Button(frame, text="Запустить подбор", command=self.run_workflow)
        self._run_button.grid(row=2, column=0, sticky="ew", pady=(6, 6))
        self._clear_button = ttk.Button(frame, text="Очистить все результаты", command=self.clear_results)
        self._clear_button.grid(row=3, column=0, sticky="ew", pady=6)
        self._status_label = ttk.Label(frame, text="", foreground="gray", wraplength=280, justify=tk.LEFT)
        self._status_label.grid(row=4, column=0, sticky="w", pady=(6, 0))
        frame.columnconfigure(0, weight=1)

    def _build_results(self, parent: ttk.Frame) -> None:
        result_frame = ttk.LabelFrame(parent, text="Лучшие решения", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("rank", "section", "mass", "stress", "deflection", "safety", "status", "model")
        self.tree = ttk.Treeview(result_frame, columns=columns, show="headings", height=16)
        headings = {
            "rank": "№",
            "section": "Сечение",
            "mass": "Масса, кг",
            "stress": "Напряжение, МПа",
            "deflection": "Прогиб, мм",
            "safety": "Запас",
            "status": "Ограничения",
            "model": "Модель",
        }
        widths = {
            "rank": 45,
            "section": 330,
            "mass": 110,
            "stress": 130,
            "deflection": 110,
            "safety": 90,
            "status": 120,
            "model": 100,
        }
        for column in columns:
            self.tree.heading(column, text=headings[column])
            self.tree.column(column, width=widths[column], anchor=tk.CENTER if column != "section" else tk.W)
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.details = tk.Text(result_frame, height=12, wrap="word")
        self.details.pack(fill=tk.BOTH, expand=False, pady=(10, 0))
        self.details.configure(state=tk.DISABLED)

    def _bind_events(self) -> None:
        self.tree.bind("<<TreeviewSelect>>", self._show_candidate_details)

    def _handle_combo_event(self, _event: object) -> None:
        self._refresh_dynamic_fields()

    def _refresh_dynamic_fields(self) -> None:
        section_type = SECTION_OPTIONS[self.section_var.get()]
        load_type = LOAD_OPTIONS[self.load_var.get()]
        self.goal_description_var.set(GOAL_DESCRIPTIONS[self.goal_var.get()])

        primary_min_label, primary_max_label, secondary_min_label, secondary_max_label = SECTION_DIMENSION_LABELS[
            section_type
        ]
        self.primary_label_var.set(primary_min_label)
        self.primary_max_label_var.set(primary_max_label)
        self.secondary_label_var.set(secondary_min_label)
        self.secondary_max_label_var.set(secondary_max_label)

        secondary_visible = section_type not in SINGLE_DIMENSION_SECTIONS
        for label_widget, entry_widget in self.secondary_row_widgets:
            if secondary_visible:
                label_widget.grid()
                entry_widget.grid()
            else:
                label_widget.grid_remove()
                entry_widget.grid_remove()

        if load_type == LoadType.UNIFORM_DISTRIBUTED:
            self.load_magnitude_label_var.set("Нагрузка q, Н/м")
            self.position_entry.configure(state=tk.DISABLED)
        elif load_type == LoadType.CENTER_POINT:
            self.load_magnitude_label_var.set("Сила, Н")
            self.position_entry.configure(state=tk.DISABLED)
        else:
            self.load_magnitude_label_var.set("Сила, Н")
            self.position_entry.configure(state=tk.NORMAL)

    def _add_labeled_entry(
        self,
        parent: ttk.Frame,
        label: str | tk.StringVar,
        variable: tk.StringVar,
        row: int,
    ) -> ttk.Entry:
        label_widget = ttk.Label(parent, textvariable=label) if isinstance(label, tk.StringVar) else ttk.Label(parent, text=label)
        label_widget.grid(row=row * 2, column=0, sticky="w")
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row * 2 + 1, column=0, sticky="ew", pady=(0, 6))
        parent.columnconfigure(0, weight=1)
        return entry

    def _add_labeled_entry_widgets(
        self,
        parent: ttk.Frame,
        label: str | tk.StringVar,
        variable: tk.StringVar,
        row: int,
    ) -> tuple[ttk.Label, ttk.Entry]:
        label_widget = ttk.Label(parent, textvariable=label) if isinstance(label, tk.StringVar) else ttk.Label(parent, text=label)
        label_widget.grid(row=row * 2, column=0, sticky="w")
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row * 2 + 1, column=0, sticky="ew", pady=(0, 6))
        parent.columnconfigure(0, weight=1)
        return label_widget, entry

    def _add_labeled_combo(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        values: list[str],
        row: int,
    ) -> ttk.Combobox:
        ttk.Label(parent, text=label).grid(row=row * 2, column=0, sticky="w")
        combo = ttk.Combobox(parent, values=values, textvariable=variable, state="readonly")
        combo.grid(row=row * 2 + 1, column=0, sticky="ew", pady=(0, 6))
        combo.bind("<<ComboboxSelected>>", self._handle_combo_event)
        parent.columnconfigure(0, weight=1)
        return combo

    def clear_results(self) -> None:
        self._release_kompas_locks()
        try:
            clear_output_structure(self.output_dir)
        except Exception as error:
            logger.exception("Failed to clear output: %s", error)
            messagebox.showerror("Ошибка", f"Не удалось очистить папку результатов:\n{error}")
            return

        for item in self.tree.get_children():
            self.tree.delete(item)
        self.current_candidates = []
        self.current_report_path = None
        self.cad_adapter = None
        self._set_details("Папка output полностью очищена: модели, отчёты и логи удалены.")

    def _release_kompas_locks(self) -> None:
        adapter = self.cad_adapter if self.cad_adapter is not None else KompasAdapter()
        try:
            released = adapter.release_output_locks(self.output_dir)
        except Exception as error:
            logger.warning("Failed to release KOMPAS locks before cleanup: %s", error)
            return
        if released:
            logger.info("Closed %s KOMPAS document(s) from output before cleanup.", len(released))

    def run_workflow(self) -> None:
        if self._optimization_running:
            return
        try:
            self._release_kompas_locks()
            request = self._build_request()
        except Exception as error:
            messagebox.showerror("Ошибка ввода", str(error))
            return

        self._set_running(True)
        thread = threading.Thread(target=self._run_in_thread, args=(request,), daemon=True)
        thread.start()

    # Thread-safety note: all mutations of shared instance attributes
    # (current_candidates, cad_adapter, etc.) happen exclusively on the Tk
    # main thread — either directly in UI callbacks or via root.after(0, …).
    # The background thread (_run_in_thread) is read-only: it receives an
    # immutable BeamRequest and communicates results only through root.after.
    def _set_running(self, running: bool) -> None:
        self._optimization_running = running
        state = tk.DISABLED if running else tk.NORMAL
        self._run_button.configure(state=state)
        self._clear_button.configure(state=state)
        self._status_label.configure(text="Вычисление... Пожалуйста, подождите." if running else "")

    def _run_in_thread(self, request: BeamRequest) -> None:
        try:
            directories = ensure_output_structure(request.output_dir)
            log_path = configure_logging(directories["logs"])
            logger.info("Workflow started")
            summary = run_optimization(request)
            self.root.after(0, self._on_optimization_done, request, directories, log_path, summary)
        except Exception as error:
            logger.exception("Workflow failed: %s", error)
            self.root.after(0, self._on_optimization_error, error)

    def _on_optimization_done(self, request: BeamRequest, directories: dict, log_path, summary) -> None:
        try:
            self.cad_adapter = KompasAdapter() if self.create_model_var.get() else None

            created_m3d = 0
            created_mock = 0
            if self.cad_adapter is not None:
                for candidate in summary.selected:
                    model_path, cad_message = self.cad_adapter.create_model(request, candidate, directories["models"])
                    candidate.model_path = model_path
                    if model_path is not None and model_path.suffix.lower() == ".m3d":
                        created_m3d += 1
                    elif model_path is not None:
                        created_mock += 1
                    logger.info("CAD result for %s: %s", candidate.section.title, cad_message)

            report_path = save_run_report(request, summary, directories["reports"])
            self.current_report_path = report_path
            self.current_candidates = summary.selected + summary.alternatives
            self._fill_tree(summary.selected, summary.alternatives)

            summary_lines = [
                f"Сгенерировано вариантов: {summary.total_generated}",
                f"Прошло ограничения: {summary.feasible_count}",
                f"Показано решений: {len(self.current_candidates)}",
                f"Итоговый отчёт: {report_path}",
                f"Лог: {log_path}",
                f"Папка результатов: {self.output_dir}",
            ]
            if self.create_model_var.get():
                summary_lines.append(f"Модели КОМПАС (.m3d): {created_m3d}")
                if created_mock:
                    summary_lines.append(
                        f"⚠ Fallback: {created_mock} модель(ей) сохранена как JSON — "
                        f"КОМПАС недоступен или не отвечает."
                    )
            else:
                summary_lines.append("Создание моделей КОМПАС отключено.")
            self._set_details("\n".join(summary_lines))

            if self.create_model_var.get() and created_mock:
                messagebox.showwarning(
                    "КОМПАС недоступен",
                    f"Не удалось создать {created_mock} .m3d файл(ов).\n"
                    f"Модели сохранены как JSON-заглушки.\n"
                    f"Убедитесь, что КОМПАС-3D запущен и доступен через COM.",
                )
        except Exception as error:
            logger.exception("Post-optimization error: %s", error)
            messagebox.showerror("Ошибка", str(error))
        finally:
            self._set_running(False)

    def _on_optimization_error(self, error: Exception) -> None:
        self._set_running(False)
        messagebox.showerror("Ошибка вычисления", str(error))

    def _build_request(self) -> BeamRequest:
        section_type = SECTION_OPTIONS[self.section_var.get()]
        load_type = LOAD_OPTIONS[self.load_var.get()]

        primary_min = self._parse_float(self.primary_min_var.get())
        primary_max = self._parse_float(self.primary_max_var.get())
        if primary_min <= 0 or primary_max <= 0:
            raise ValueError("Размеры сечения должны быть положительными.")
        if primary_min > primary_max:
            raise ValueError("Минимальный основной размер не может превышать максимальный.")

        secondary_range = None
        if section_type not in SINGLE_DIMENSION_SECTIONS:
            secondary_min = self._parse_float(self.secondary_min_var.get())
            secondary_max = self._parse_float(self.secondary_max_var.get())
            if secondary_min <= 0 or secondary_max <= 0:
                raise ValueError("Размеры сечения должны быть положительными.")
            if secondary_min > secondary_max:
                raise ValueError("Минимальный вторичный размер не может превышать максимальный.")
            secondary_range = NumericRange(
                min_value=secondary_min,
                max_value=secondary_max,
                step=self._extreme_step(self.secondary_min_var.get(), self.secondary_max_var.get()),
            )

        search_space = SectionSearchSpace(
            primary=NumericRange(
                min_value=primary_min,
                max_value=primary_max,
                step=self._extreme_step(self.primary_min_var.get(), self.primary_max_var.get()),
            ),
            secondary=secondary_range,
        )

        load_case = LoadCase(
            load_type=load_type,
            magnitude=self._parse_float(self.load_magnitude_var.get()),
            position_m=(
                self._parse_float(self.load_position_var.get()) if load_type == LoadType.POINT_AT_POSITION else None
            ),
        )

        constraints = Constraints(
            max_stress_pa=self._optional_float(self.max_stress_var.get(), scale=1e6),
            max_deflection_m=self._optional_float(self.max_deflection_var.get(), scale=1e-3),
            min_safety_factor=self._optional_float(self.min_safety_var.get()),
            max_primary_dimension_mm=self._optional_float(self.max_primary_dim_var.get()),
            max_secondary_dimension_mm=self._optional_float(self.max_secondary_dim_var.get()),
        )

        request = BeamRequest(
            length_m=self._parse_float(self.length_var.get()),
            section_type=section_type,
            material=get_material_by_title(self.material_var.get()),
            support_type=SUPPORT_OPTIONS[self.support_var.get()],
            load_case=load_case,
            search_space=search_space,
            constraints=constraints,
            optimization_goal=GOAL_OPTIONS[self.goal_var.get()],
            top_n=int(self._parse_float(self.top_n_var.get())),
            output_dir=self.output_dir,
        )

        if request.length_m <= 0:
            raise ValueError("Длина балки должна быть больше нуля.")
        if request.top_n <= 0:
            raise ValueError("Количество лучших вариантов должно быть положительным.")
        if (
            request.load_case.load_type == LoadType.POINT_AT_POSITION
            and request.load_case.position_m is not None
            and not (0.0 < request.load_case.position_m <= request.length_m)
        ):
            raise ValueError(
                "Координата сосредоточенной силы должна быть в пределах (0, L]. "
                "Для консоли допустима нагрузка на свободном конце (x = L)."
            )
        return request

    def _fill_tree(self, selected, alternatives) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)

        rows = [("best", candidate) for candidate in selected] + [("alt", candidate) for candidate in alternatives]
        for index, (kind, candidate) in enumerate(rows, start=1):
            status = "OK" if candidate.is_feasible else "Нет"
            model_text = "Да" if candidate.model_path and candidate.model_path.suffix.lower() == ".m3d" else "Нет"
            tag = "selected" if kind == "best" else "alternative"
            self.tree.insert(
                "",
                tk.END,
                iid=str(index - 1),
                values=(
                    index,
                    candidate.section.title,
                    f"{candidate.analysis.mass_kg:.2f}",
                    f"{candidate.analysis.max_stress_pa / 1e6:.2f}",
                    f"{candidate.analysis.max_deflection_m * 1000.0:.2f}",
                    f"{candidate.analysis.safety_factor:.2f}",
                    status,
                    model_text,
                ),
                tags=(tag,),
            )
        self.tree.tag_configure("selected", background="#eef7e8")
        self.tree.tag_configure("alternative", background="#f7f7f7")

    def _show_candidate_details(self, _event: object) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        candidate = self.current_candidates[int(selection[0])]
        lines = [
            f"Сечение: {candidate.section.title}",
            f"Размеры: {candidate.section.dimensions_mm}",
            f"Масса: {candidate.analysis.mass_kg:.3f} кг",
            f"Максимальный момент: {candidate.analysis.max_moment_nm:.3f} Н*м",
            f"Максимальное напряжение: {candidate.analysis.max_stress_pa / 1e6:.3f} МПа",
            f"Максимальный прогиб: {candidate.analysis.max_deflection_m * 1000.0:.3f} мм",
            f"Коэффициент запаса: {candidate.analysis.safety_factor:.3f}",
            f"Ограничения: {'пройдены' if candidate.is_feasible else 'не пройдены'}",
            f"Нарушения: {', '.join(candidate.violations) if candidate.violations else '-'}",
            f"Модель КОМПАС: {candidate.model_path if candidate.model_path else 'не создавалась'}",
        ]
        if self.current_report_path:
            lines.append(f"Итоговый отчёт запуска: {self.current_report_path}")
        self._set_details("\n".join(lines))

    def _set_details(self, text: str) -> None:
        self.details.configure(state=tk.NORMAL)
        self.details.delete("1.0", tk.END)
        self.details.insert("1.0", text)
        self.details.configure(state=tk.DISABLED)

    @staticmethod
    def _parse_float(raw_value: str) -> float:
        return float(raw_value.strip().replace(",", "."))

    @classmethod
    def _extreme_step(cls, min_value_raw: str, max_value_raw: str) -> float:
        min_value = cls._parse_float(min_value_raw)
        max_value = cls._parse_float(max_value_raw)
        delta = max_value - min_value
        if delta <= 0:
            return 1.0
        return delta / DEFAULT_RANGE_SUBDIVISIONS

    @staticmethod
    def _optional_float(raw_value: str, scale: float = 1.0) -> float | None:
        cleaned = raw_value.strip()
        if not cleaned:
            return None
        return float(cleaned.replace(",", ".")) * scale


def run_app() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    BeamOptimizerApp(root)
    root.mainloop()
