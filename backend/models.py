"""
SQLAlchemy ORM 모델.

restaurants 테이블 = 기존 데이터 파이프라인에서 만든 target_df
(구, 소재지면적, 업태구분명, 동종업체수, 면적당종사자수, 구, 총종사자수, label)를
1/3/5년 기준으로 각각 라벨링해 저장한 테이블.

주의: 원본 파이프라인의 label은 "관측 시점 기준 생존 여부" 하나였지만,
화면 요건상 1/3/5년 각각의 예측/실측 비교가 필요하므로
init_db.py에서 연차별 데이터셋(df_1years/df_3years/df_5years, 기존 get_yearly_data 결과)을
받아 label_1y/label_3y/label_5y 세 컬럼으로 병합해 적재한다.
"""
from sqlalchemy import Column, Float, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from database import Base


class Gu(Base):
    __tablename__ = "gu"

    gu_id = Column(Integer, primary_key=True)
    gu_name = Column(String, nullable=False, unique=True)

    dongs = relationship(
        "Dong",
        back_populates="gu",
        cascade="all, delete-orphan"
    )


class Dong(Base):
    __tablename__ = "dong"

    # dong_id가 전체에서 고유하다면 단일 PK로 설정
    dong_id = Column(Integer, primary_key=True)

    gu_id = Column(
        Integer,
        ForeignKey("gu.gu_id"),
        nullable=False,
        index=True
    )
    dong_name = Column(String, nullable=False, index=True)

    gu = relationship("Gu", back_populates="dongs")

    businesses = relationship(
        "Business",
        back_populates="dong"
    )


class Business(Base):
    __tablename__ = "business"

    business_id = Column(Integer, primary_key=True, autoincrement=True)

    permit_date = Column(String, nullable=False, index=True)
    business_status = Column(String, nullable=False, index=True)
    closure_date = Column(String, nullable=True, index=True)

    site_area = Column(Integer, nullable=True)
    business_name = Column(String, nullable=False, index=True)
    business_type = Column(String, nullable=False, index=True)
    lot_address = Column(String, nullable=False, index=True)
    last_modified_at = Column(String, nullable=False, index=True)

    dong_id = Column(
        Integer,
        ForeignKey("dong.dong_id"),
        nullable=False,
        index=True
    )

    total_workers = Column(Integer, nullable=True)

    label_1 = Column(Integer, nullable=True)
    label_3 = Column(Integer, nullable=True)
    label_5 = Column(Integer, nullable=True)

    dong = relationship("Dong", back_populates="businesses")