from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from beam_optimizer.models import BeamRequest, CandidateResult, SectionType

logger = logging.getLogger(__name__)

KOMPAS_API7_PROGID = "KOMPAS.Application.7"
DOCUMENT_PART = 4
O3D_CYLINDER = 663
O3D_BLOCK_BY_SIZES = 668
BOOLEAN_DIFFERENCE = 2
OPERATION_NEW_BODY = 1


class KompasAdapter:
    def __init__(self) -> None:
        self._kompas = None
        self._win32_client = None

    def create_model(self, request: BeamRequest, candidate: CandidateResult, output_dir: Path) -> tuple[Path | None, str]:
        try:
            return self._create_live_model(request, candidate, output_dir)
        except Exception as error:
            logger.exception("KOMPAS live export failed, fallback to mock: %s", error)
        return self._create_mock_model(request, candidate, output_dir)

    def _create_live_model(self, request: BeamRequest, candidate: CandidateResult, output_dir: Path) -> tuple[Path, str]:
        kompas = self._ensure_kompas_application()
        document_3d = self._create_document3d(kompas)
        container = self._get_model_container(document_3d)
        self._apply_material(container, request.material.title, request.material.density_kg_m3)
        self._build_beam_body(container, request, candidate)

        model_path = self._resolve_model_path(output_dir, _slug(candidate.section.title))
        document_3d.SaveAs(str(model_path))
        time.sleep(0.5)
        if not model_path.exists():
            raise RuntimeError("KOMPAS не сохранил ожидаемый файл модели.")
        return model_path, "Сохранена реальная 3D-модель КОМПАС."

    def _apply_material(self, container, material_name: str, density_kg_m3: float) -> None:
        try:
            part = self._cast(container, "IPart7")
        except Exception as error:
            logger.warning("Cannot cast model container to IPart7 to set material: %s", error)
            return

        try:
            part.SetMaterial(str(material_name), float(density_kg_m3))
            logger.info("KOMPAS material set: %s (density=%s)", material_name, density_kg_m3)
            return
        except Exception as set_error:
            logger.warning("IPart7.SetMaterial failed: %s", set_error)

        applied = False
        try:
            part.Material = str(material_name)
            applied = True
        except Exception:
            pass
        try:
            part.Density = float(density_kg_m3)
            applied = True
        except Exception:
            pass
        if applied:
            logger.info("KOMPAS material set via property fallback: %s (density=%s)", material_name, density_kg_m3)
        else:
            logger.warning("Failed to apply KOMPAS material. Document may keep default material.")

    def _create_mock_model(self, request: BeamRequest, candidate: CandidateResult, output_dir: Path) -> tuple[Path, str]:
        model_path = output_dir / f"{_slug(candidate.section.title)}.kompas.json"
        payload = {
            "mode": "mock",
            "length_m": request.length_m,
            "material": request.material.title,
            "section_type": candidate.section.section_type.value,
            "dimensions_mm": candidate.section.dimensions_mm,
            "support_type": request.support_type.value,
            "load_type": request.load_case.load_type.value,
            "load_magnitude": request.load_case.magnitude,
        }
        model_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return model_path, "Создан mock-файл вместо модели КОМПАС."

    @staticmethod
    def _resolve_model_path(output_dir: Path, slug: str) -> Path:
        base_path = (output_dir / f"{slug}.m3d").resolve()
        if not base_path.exists():
            return base_path
        try:
            base_path.unlink()
            return base_path
        except OSError:
            pass

        index = 1
        while True:
            candidate = (output_dir / f"{slug}_{index}.m3d").resolve()
            if not candidate.exists():
                return candidate
            try:
                candidate.unlink()
                return candidate
            except OSError:
                index += 1

    def _ensure_kompas_application(self):
        if self._kompas is not None:
            return self._kompas

        try:
            import win32com.client as win32_client
        except ImportError as error:
            raise RuntimeError("Для создания модели в КОМПАС нужен пакет pywin32.") from error

        try:
            kompas = win32_client.gencache.EnsureDispatch(KOMPAS_API7_PROGID)
        except Exception as error:
            raise RuntimeError(
                "Не удалось подключиться к COM-серверу КОМПАС API 7. Проверьте установленный КОМПАС-3D и регистрацию COM."
            ) from error

        kompas.Visible = True
        self._kompas = kompas
        self._win32_client = win32_client
        logger.info("Connected to KOMPAS COM server via %s", KOMPAS_API7_PROGID)
        return kompas

    def _create_document3d(self, kompas):
        document = kompas.Documents.Add(DOCUMENT_PART, True)
        return self._cast(document, "IKompasDocument3D")

    def _get_model_container(self, document_3d):
        top_part = document_3d.TopPart
        if top_part is None:
            raise RuntimeError("Не удалось получить верхний компонент 3D-документа КОМПАС.")
        return self._cast(top_part, "IModelContainer")

    def _build_beam_body(self, container, request: BeamRequest, candidate: CandidateResult) -> None:
        section_type = candidate.section.section_type
        length_mm = request.length_m * 1000.0

        if section_type == SectionType.RECTANGULAR:
            self._create_block_body(
                container,
                length_mm=length_mm,
                width_mm=candidate.section.dimensions_mm["width"],
                height_mm=candidate.section.dimensions_mm["height"],
            )
            return

        if section_type == SectionType.SQUARE:
            side_mm = candidate.section.dimensions_mm["side"]
            self._create_block_body(
                container,
                length_mm=length_mm,
                width_mm=side_mm,
                height_mm=side_mm,
            )
            return

        if section_type == SectionType.SOLID_ROUND:
            self._create_cylinder_body(
                container,
                length_mm=length_mm,
                diameter_mm=candidate.section.dimensions_mm["diameter"],
            )
            return

        if section_type == SectionType.ROUND_TUBE:
            self._create_round_tube_body(
                container,
                length_mm=length_mm,
                outer_diameter_mm=candidate.section.dimensions_mm["outer_diameter"],
                thickness_mm=candidate.section.dimensions_mm["thickness"],
            )
            return

        raise RuntimeError(f"Сечение {section_type.value} пока не поддержано для live-экспорта в КОМПАС.")

    def _create_block_body(self, container, length_mm: float, width_mm: float, height_mm: float) -> Any:
        block = self._cast(container.ElementaryBodies.Add(O3D_BLOCK_BY_SIZES), "IBlockBySizes")
        block.OperationResult = OPERATION_NEW_BODY
        block.Length = length_mm
        block.Width = width_mm
        block.SetHeight(True, height_mm)
        if not block.Update():
            raise RuntimeError("Не удалось создать твердотельную балку прямоугольного сечения в КОМПАС.")
        return block

    def _create_cylinder_body(self, container, length_mm: float, diameter_mm: float) -> Any:
        cylinder = self._cast(container.ElementaryBodies.Add(O3D_CYLINDER), "ICylinder")
        cylinder.OperationResult = OPERATION_NEW_BODY
        cylinder.Diameter = diameter_mm
        cylinder.SetHeight(True, length_mm)
        if not cylinder.Update():
            raise RuntimeError("Не удалось создать цилиндрическое тело в КОМПАС.")
        return cylinder

    def _create_round_tube_body(
        self,
        container,
        length_mm: float,
        outer_diameter_mm: float,
        thickness_mm: float,
    ) -> None:
        inner_diameter_mm = outer_diameter_mm - 2.0 * thickness_mm
        if inner_diameter_mm <= 0:
            raise RuntimeError("Для круглой трубы внутренний диаметр должен быть больше нуля.")

        outer_body = self._create_cylinder_body(
            container=container,
            length_mm=length_mm,
            diameter_mm=outer_diameter_mm,
        )
        inner_body = self._create_cylinder_body(
            container=container,
            length_mm=length_mm,
            diameter_mm=inner_diameter_mm,
        )

        boolean = self._cast(container.Booleans.Add(), "IBoolean")
        boolean.BooleanType = BOOLEAN_DIFFERENCE
        boolean.BaseObject = outer_body
        boolean.ModifyObjects = [inner_body]
        boolean.SaveCopyBaseObject = False
        boolean.SaveCopyModifyObjects = False
        if not boolean.Update():
            raise RuntimeError("Не удалось построить круглую трубу в КОМПАС.")

    def _cast(self, obj: Any, interface_name: str):
        if self._win32_client is None:
            raise RuntimeError("COM-клиент pywin32 не инициализирован.")
        return self._win32_client.CastTo(obj, interface_name)


    def release_output_locks(self, output_dir: Path) -> list[Path]:
        output_root = output_dir.resolve()
        kompas = self._kompas or self._try_attach_to_running_kompas()
        if kompas is None:
            return []

        try:
            documents = kompas.Documents
            count = int(getattr(documents, "Count", 0))
        except Exception as error:
            logger.warning("Cannot access KOMPAS documents to release locks: %s", error)
            return []

        released: list[Path] = []
        for index in range(count, 0, -1):
            try:
                document = documents.Item(index)
            except Exception:
                continue

            path_value = self._read_document_path(document)
            if not path_value:
                continue

            try:
                document_path = Path(path_value).resolve()
            except Exception:
                continue

            if output_root == document_path or output_root in document_path.parents:
                if self._close_document(document):
                    released.append(document_path)

        return released

    def _try_attach_to_running_kompas(self):
        if self._kompas is not None:
            return self._kompas
        try:
            import win32com.client as win32_client
        except ImportError:
            return None

        try:
            kompas = win32_client.GetActiveObject(KOMPAS_API7_PROGID)
        except Exception:
            return None

        self._kompas = kompas
        self._win32_client = win32_client
        return kompas

    @staticmethod
    def _read_document_path(document) -> str:
        for attribute in ("PathName", "FullName"):
            try:
                value = getattr(document, attribute, None)
            except Exception:
                continue
            if value:
                return str(value)
        return ""

    @staticmethod
    def _close_document(document) -> bool:
        close_signatures = ((False,), (0,), (), (True,), (1,))
        for args in close_signatures:
            try:
                document.Close(*args)
                return True
            except TypeError:
                continue
            except Exception:
                continue
        return False


def _slug(text: str) -> str:
    normalized = "".join(character.lower() if character.isalnum() else "_" for character in text)
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")
