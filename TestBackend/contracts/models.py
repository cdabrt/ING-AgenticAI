from __future__ import annotations

import enum
from typing import List, Optional
from sqlalchemy import ForeignKey, Enum, LargeBinary
from sqlalchemy.orm import relationship, DeclarativeBase, Mapped, mapped_column

class RequirementType(str, enum.Enum):
    BUSINESS = "business"
    DATA = "data"

class SourceType(str, enum.Enum):
    DOCUMENT = "document"
    ONLINE = "online"

class Base(DeclarativeBase):
    pass


class RequirementBundle(Base):
    __tablename__ = "RequirementBundles"
    id: Mapped[int] = mapped_column(primary_key=True, index=True, autoincrement=True)
    document: Mapped[str] = mapped_column(nullable=False)
    document_type: Mapped[str] = mapped_column(nullable=False)

    # Relationships
    requirements: Mapped[List[RequirementItem]] = relationship(back_populates="bundle", cascade="all, delete-orphan")
    assumptions: Mapped[List[Assumption]] = relationship(back_populates="bundle", cascade="all, delete-orphan")


class RequirementItem(Base):
    __tablename__ = "RequirementItems"
    id: Mapped[int] = mapped_column(primary_key=True, index=True, autoincrement=True)
    req_id: Mapped[str] = mapped_column(nullable=False)
    description: Mapped[str] = mapped_column(nullable=False)
    rationale: Mapped[str] = mapped_column(nullable=False)
    type: Mapped[RequirementType] = mapped_column(Enum(RequirementType), nullable=False)

    # Relationships
    sources: Mapped[List[Source]] = relationship(back_populates="requirement", cascade="all, delete-orphan")

    # Foreign Keys
    bundle_id: Mapped[int] = mapped_column(ForeignKey("RequirementBundles.id"), nullable=False)
    bundle: Mapped[RequirementBundle] = relationship(back_populates="requirements")


class Assumption(Base):
    __tablename__ = "Assumptions"
    id: Mapped[int] = mapped_column(primary_key=True, index=True, autoincrement=True)
    text: Mapped[str] = mapped_column(nullable=False)

    # Foreign Keys
    bundle_id: Mapped[int] = mapped_column(ForeignKey("RequirementBundles.id"), nullable=False)
    bundle: Mapped[RequirementBundle] = relationship(back_populates="assumptions")
    
class Source(Base):
    __tablename__ = "Sources"
    id: Mapped[int] = mapped_column(primary_key=True, index=True, autoincrement=True)
    source_type: Mapped[SourceType] = mapped_column(Enum(SourceType), nullable=False)
    reference: Mapped[str] = mapped_column(nullable=False)
    
    # Foreign Keys
    requirement_id: Mapped[int] = mapped_column(ForeignKey("RequirementItems.id"), nullable=False)
    requirement: Mapped[RequirementItem] = relationship(back_populates="sources")
    
    pdf_id: Mapped[Optional[int]] = mapped_column(ForeignKey("PDFDocuments.id"), nullable=True)
    pdf: Mapped[Optional[PDFDocument]] = relationship(back_populates="sources")


class PDFDocument(Base):
    __tablename__ = "PDFDocuments"
    id: Mapped[int] = mapped_column(primary_key=True, index=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(nullable=False)
    pdf_data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    
    # Relationships
    sources: Mapped[List[Source]] = relationship(back_populates="pdf", cascade="all, delete-orphan")