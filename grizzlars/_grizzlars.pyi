####################    <generated_from:grizzlars_shim.h>    ####################



class GrizzlarFrame:
    def __init__(self) -> None:
        pass

    # ── construction / loading ────────────────────────────────────────────
    # Taken by value (not const&): nanobind already materializes a fresh
    # std::vector<T> converting from the Python list/array regardless, so
    # taking it by value costs nothing extra and lets the implementation
    # std::move it straight into hmdf's storage instead of copying again.
    def load_index(self, indices: List[int]) -> None:
        pass
    def load_column_double(self, name: str, values: List[float]) -> None:
        pass
    def load_column_int64(self, name: str, values: List[int]) -> None:
        pass
    def load_column_bool(self, name: str, values: List[int]) -> None:
        pass
    def load_column_string(self, name: str, values: List[str]) -> None:
        pass

    # ── accessors ──────────────────────────────────────────────────────────
    def deep_copy(self) -> GrizzlarFrame:
        pass
    def get_index(self) -> List[int]:
        pass
    def columns(self) -> List[str]:
        pass
    def shape(self) -> Tuple[int, int]:
        pass
    def has_column(self, name: str) -> bool:
        pass
    def col_type(self, name: str) -> str:
        pass
    def drop_column(self, name: str) -> None:
        pass

    def get_column_double(self, name: str) -> List[float]:
        pass
    def get_column_int64(self, name: str) -> List[int]:
        pass
    def get_column_bool(self, name: str) -> List[int]:
        pass
    def get_column_string(self, name: str) -> List[str]:
        pass

    # ── scalar statistics (real hmdf visitors) ────────────────────────────
    def mean(self, col: str) -> float:
        pass
    def std_dev(self, col: str) -> float:
        pass
    def sum(self, col: str) -> float:
        pass
    def col_min(self, col: str) -> float:
        pass
    def col_max(self, col: str) -> float:
        pass
    def quantile(self, col: str, q: float) -> float:
        pass
    def corr(self, col1: str, col2: str) -> float:
        pass
    def cov(self, col1: str, col2: str) -> float:
        pass
    def skew_col(self, col: str) -> float:
        pass
    def kurt_col(self, col: str) -> float:
        pass
    def mode_col_double(self, col: str) -> List[float]:
        pass
    def mode_col_int64(self, col: str) -> List[int]:
        pass
    def mode_col_string(self, col: str) -> List[str]:
        pass

    def nunique(self, col: str) -> int:
        pass
    def n_missing(self, col: str) -> int:
        pass
    def count(self, col: str) -> int:
        pass
    def unique_double(self, col: str) -> List[float]:
        pass
    def unique_int64(self, col: str) -> List[int]:
        pass
    def unique_string(self, col: str) -> List[str]:
        pass
    def value_counts_double(self, col: str) -> GrizzlarFrame:
        pass
    def value_counts_int64(self, col: str) -> GrizzlarFrame:
        pass
    def value_counts_string(self, col: str) -> GrizzlarFrame:
        pass
    def describe(self) -> Dict[str, Dict[str, float]]:
        pass

    # ── sorting / row selection / filtering ───────────────────────────────
    def sort_by(self, by: str, ascending: bool) -> GrizzlarFrame:
        pass
    def sort_index(self, ascending: bool) -> GrizzlarFrame:
        pass
    def iloc(self, start: int, stop: int) -> GrizzlarFrame:
        pass
    def take_rows(self, positions: List[int]) -> GrizzlarFrame:
        pass
    def select_columns(self, names: List[str]) -> GrizzlarFrame:
        pass
    def filter_by_mask_list(self, mask: List[int]) -> GrizzlarFrame:
        pass
    def filter_col_scalar_double(self, col: str, op: str, scalar: float) -> GrizzlarFrame:
        pass
    def compare_col_scalar_double(self, col: str, op: str, scalar: float) -> List[int]:
        pass

    # ── groupby / join / concat ────────────────────────────────────────────
    def groupby_agg(
        self,
        by_col: str,
        agg_cols: List[str],
        agg_funcs: List[str]
        ) -> GrizzlarFrame:
        pass
    def join_by_index(self, rhs: GrizzlarFrame, how: str) -> GrizzlarFrame:
        pass
    def concat_frame(self, other: GrizzlarFrame) -> GrizzlarFrame:
        pass

    # ── missing data ───────────────────────────────────────────────────────
    def fillna_double(self, col: str, value: float) -> None:
        pass
    def fillna_string(self, col: str, value: str) -> None:
        pass
    def ffill_col(self, col: str) -> None:
        pass
    def bfill_col(self, col: str) -> None:
        pass
    def drop_na(self, col: str) -> GrizzlarFrame:
        pass
    def drop_duplicates(self, col: str) -> GrizzlarFrame:
        pass

    # ── window functions ───────────────────────────────────────────────────
    def cumulative(self, col: str, func: str) -> List[float]:
        pass
    def shift_col(self, col: str, n: int) -> List[float]:
        pass
    def pct_change(self, col: str) -> List[float]:
        pass
    def rolling(self, col: str, window: int, func: str) -> List[float]:
        pass

    # ── CSV I/O ────────────────────────────────────────────────────────────
    # A straightforward sequential reader/writer (not memory-mapped or
    # multithreaded, unlike the previous hand-rolled implementation) — hmdf's
    # real read()/write() are schema/index-centric in a way that doesn't fit
    # grizzlars' "no index column given" case cleanly, so this trades the
    # old CSV-loading performance edge for a simple, predictable path.
    def to_csv(self, path: str, write_index: bool) -> None:
        pass
    @staticmethod
    def read_csv_native(path: str, index_col_name: str) -> GrizzlarFrame:
        pass

    # ── data cleaning ──────────────────────────────────────────────────────
    def clip_col(self, col: str, lower: float, upper: float) -> None:
        pass
    def round_col(self, col: str, decimals: int) -> None:
        pass
    def abs_col(self, col: str) -> None:
        pass
    def rename_col(self, old_name: str, new_name: str) -> None:
        pass
    def astype_col(self, col: str, target_type: str) -> None:
        pass
    def replace_col_double(self, col: str, from_: List[float], to: List[float]) -> None:
        pass
    def replace_col_string(self, col: str, from_: List[str], to: List[str]) -> None:
        pass
    def isin_col_double(self, col: str, values: List[float]) -> List[int]:
        pass
    def isin_col_string(self, col: str, values: List[str]) -> List[int]:
        pass
    def duplicated_rows(self, cols: List[str], keep: str) -> List[int]:
        pass
    def diff_col(self, col: str, periods: int) -> List[float]:
        pass

    # ── boolean-frame / reduction ops ──────────────────────────────────────
    def isna_frame(self) -> GrizzlarFrame:
        pass
    def notna_frame(self) -> GrizzlarFrame:
        pass
    def where_frame(self, cond_frame: GrizzlarFrame, fill_val: float) -> GrizzlarFrame:
        pass
    def arith_scalar(self, op: str, scalar: float) -> GrizzlarFrame:
        pass
    def arith_frame_op(self, op: str, other: GrizzlarFrame) -> GrizzlarFrame:
        pass
    def compare_scalar(self, op: str, scalar: float) -> GrizzlarFrame:
        pass
    def reduce_all(self, func: str) -> GrizzlarFrame:
        pass

    # ── reshaping ───────────────────────────────────────────────────────────
    def transpose_frame(self) -> GrizzlarFrame:
        pass
    def set_index_col(self, col: str, drop: bool) -> GrizzlarFrame:
        pass
    def reset_index_frame(self, drop: bool) -> GrizzlarFrame:
        pass
    def melt_frame(
        self,
        id_cols: List[str],
        val_cols: List[str],
        var_name: str,
        value_name: str
        ) -> GrizzlarFrame:
        pass

####################    </generated_from:grizzlars_shim.h>    ####################
