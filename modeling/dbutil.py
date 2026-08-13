from pathlib import Path
import sqlite3
import pandas as pd


REQUIRED_COLUMNS = [
    "인허가일자",
    "영업상태명",
    "폐업일자",
    "소재지면적",
    "사업장명",
    "업태구분명",
    "지번주소",
    "최종수정시점",
    "구",
    "동",
    "총종사자수",
    "label_1",
    "label_3",
    "label_5",
]


def clean_text(series: pd.Series) -> pd.Series:
    """문자열 공백 제거, 빈 문자열을 결측치로 변환"""
    series = series.astype("string").str.strip()
    return series.replace("", pd.NA)


def to_date_string(series: pd.Series) -> pd.Series:
    """YYYY-MM-DD 형식의 날짜 문자열로 변환"""
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


def to_datetime_string(series: pd.Series) -> pd.Series:
    """YYYY-MM-DD HH:MM:SS 형식의 일시 문자열로 변환"""
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")


def validate_labels(df: pd.DataFrame) -> None:
    """label 값이 0, 1 또는 NULL인지 검증"""
    for label_col in ["label_1", "label_3", "label_5"]:
        values = pd.to_numeric(df[label_col], errors="coerce")
        invalid_values = values.dropna()[~values.dropna().isin([0, 1])]

        if not invalid_values.empty:
            raise ValueError(
                f"{label_col}에는 0, 1, 결측치만 들어갈 수 있습니다. "
                f"잘못된 값 예시: {invalid_values.iloc[:5].tolist()}"
            )


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """SQLite 저장 전 데이터 전처리"""
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"다음 필수 컬럼이 없습니다: {sorted(missing_columns)}"
        )

    data = df[REQUIRED_COLUMNS].copy()

    validate_labels(data)

    text_columns = [
        "영업상태명",
        "사업장명",
        "업태구분명",
        "지번주소",
        "구",
        "동",
    ]

    for col in text_columns:
        data[col] = clean_text(data[col])

    data["인허가일자"] = to_date_string(data["인허가일자"])
    data["폐업일자"] = to_date_string(data["폐업일자"])
    data["최종수정시점"] = to_datetime_string(data["최종수정시점"])

    data["소재지면적"] = pd.to_numeric(
        data["소재지면적"],
        errors="coerce"
    )

    data["총종사자수"] = pd.to_numeric(
        data["총종사자수"],
        errors="coerce"
    ).astype("Int64")

    for col in ["label_1", "label_3", "label_5"]:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        ).astype("Int64")

    required_not_null = [
        "인허가일자",
        "업태구분명",
        "구",
        "동",
    ]

    original_count = len(data)
    data = data.dropna(subset=required_not_null).copy()
    removed_count = original_count - len(data)

    if removed_count > 0:
        print(f"필수값 누락으로 {removed_count:,}개 행을 제거했습니다.")

    return data


def create_tables(conn: sqlite3.Connection) -> None:
    """기존 테이블 삭제 후 새 테이블 생성"""
    conn.execute("PRAGMA foreign_keys = ON")

    conn.executescript(
        """
        DROP TABLE IF EXISTS business;
        DROP TABLE IF EXISTS dong;
        DROP TABLE IF EXISTS gu;

        CREATE TABLE gu (
            gu_id       INTEGER PRIMARY KEY,
            gu_name     TEXT NOT NULL UNIQUE
        );

        CREATE TABLE dong (
            dong_id     INTEGER PRIMARY KEY,
            gu_id       INTEGER NOT NULL,
            dong_name   TEXT NOT NULL,

            FOREIGN KEY (gu_id) REFERENCES gu(gu_id),
            UNIQUE (gu_id, dong_name)
        );

        CREATE TABLE business (
            business_id         INTEGER PRIMARY KEY AUTOINCREMENT,

            permit_date         TEXT NOT NULL,
            business_status     TEXT,
            closure_date        TEXT,

            site_area           REAL,
            business_name       TEXT,
            business_type       TEXT NOT NULL,

            lot_address         TEXT,
            last_modified_at    TEXT,

            dong_id             INTEGER NOT NULL,
            total_workers       INTEGER,

            label_1             INTEGER CHECK (label_1 IN (0, 1) OR label_1 IS NULL),
            label_3             INTEGER CHECK (label_3 IN (0, 1) OR label_3 IS NULL),
            label_5             INTEGER CHECK (label_5 IN (0, 1) OR label_5 IS NULL),

            FOREIGN KEY (dong_id) REFERENCES dong(dong_id)
        );

        CREATE INDEX idx_dong_gu
        ON dong (gu_id, dong_name);

        CREATE INDEX idx_business_dong_type
        ON business (dong_id, business_type);

        CREATE INDEX idx_business_dong_type_permit
        ON business (dong_id, business_type, permit_date);

        CREATE INDEX idx_business_competitor_lookup
        ON business (dong_id, business_type, permit_date, closure_date);
        """
    )


def to_sql_value(value):
    """pandas 결측치를 SQLite NULL로 변환"""
    if pd.isna(value):
        return None

    if isinstance(value, (int, float, str)):
        return value

    return str(value)


def insert_data(conn: sqlite3.Connection, data: pd.DataFrame) -> None:
    """구, 동, 사업장 데이터를 순서대로 삽입"""

    # 1. 구 테이블 삽입
    gu_names = sorted(data["구"].dropna().unique().tolist())

    conn.executemany(
        "INSERT INTO gu (gu_name) VALUES (?)",
        [(gu_name,) for gu_name in gu_names]
    )

    gu_map = {
        gu_name: gu_id
        for gu_id, gu_name in conn.execute(
            "SELECT gu_id, gu_name FROM gu"
        ).fetchall()
    }

    # 2. 동 테이블 삽입
    dong_data = (
        data[["구", "동"]]
        .drop_duplicates()
        .sort_values(["구", "동"])
        .copy()
    )

    dong_rows = [
        (gu_map[row["구"]], row["동"])
        for _, row in dong_data.iterrows()
    ]

    conn.executemany(
        """
        INSERT INTO dong (gu_id, dong_name)
        VALUES (?, ?)
        """,
        dong_rows
    )

    # 3. 구 + 동 조합에서 dong_id를 찾기 위한 매핑
    dong_map_df = pd.read_sql_query(
        """
        SELECT
            g.gu_name AS 구,
            d.dong_name AS 동,
            d.dong_id
        FROM dong AS d
        JOIN gu AS g
          ON d.gu_id = g.gu_id
        """,
        conn
    )

    data = data.merge(
        dong_map_df,
        on=["구", "동"],
        how="left",
        validate="many_to_one"
    )

    if data["dong_id"].isna().any():
        raise ValueError("일부 행의 dong_id를 생성하지 못했습니다.")

    # 4. 사업장 테이블 삽입
    insert_columns = [
        "인허가일자",
        "영업상태명",
        "폐업일자",
        "소재지면적",
        "사업장명",
        "업태구분명",
        "지번주소",
        "최종수정시점",
        "dong_id",
        "총종사자수",
        "label_1",
        "label_3",
        "label_5",
    ]

    business_rows = []

    for row in data[insert_columns].to_dict(orient="records"):
        business_rows.append(
            (
                to_sql_value(row["인허가일자"]),
                to_sql_value(row["영업상태명"]),
                to_sql_value(row["폐업일자"]),
                to_sql_value(row["소재지면적"]),
                to_sql_value(row["사업장명"]),
                to_sql_value(row["업태구분명"]),
                to_sql_value(row["지번주소"]),
                to_sql_value(row["최종수정시점"]),
                int(row["dong_id"]),
                to_sql_value(row["총종사자수"]),
                to_sql_value(row["label_1"]),
                to_sql_value(row["label_3"]),
                to_sql_value(row["label_5"]),
            )
        )

    conn.executemany(
        """
        INSERT INTO business (
            permit_date,
            business_status,
            closure_date,
            site_area,
            business_name,
            business_type,
            lot_address,
            last_modified_at,
            dong_id,
            total_workers,
            label_1,
            label_3,
            label_5
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        business_rows
    )


def build_sqlite_database(
    df: pd.DataFrame,
    db_path: str = "business.db"
) -> None:
    """DataFrame으로부터 SQLite 데이터베이스 생성"""

    data = prepare_data(df)
    db_path = Path(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")

        create_tables(conn)
        insert_data(conn, data)

        gu_count = conn.execute(
            "SELECT COUNT(*) FROM gu"
        ).fetchone()[0]

        dong_count = conn.execute(
            "SELECT COUNT(*) FROM dong"
        ).fetchone()[0]

        business_count = conn.execute(
            "SELECT COUNT(*) FROM business"
        ).fetchone()[0]

    print(f"DB 생성 완료: {db_path.resolve()}")
    print(f"구 개수: {gu_count:,}")
    print(f"동 개수: {dong_count:,}")
    print(f"사업장 개수: {business_count:,}")