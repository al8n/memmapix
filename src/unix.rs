use std::{
  fs::File,
  io,
  mem::ManuallyDrop,
  os::unix::io::{FromRawFd, RawFd},
  ptr,
  sync::atomic::{AtomicUsize, Ordering},
};

use rustix::{
  fd::BorrowedFd,
  mm::{MapFlags, MprotectFlags, MsyncFlags, ProtFlags},
};

#[cfg(any(
  all(target_os = "linux", not(target_arch = "mips")),
  target_os = "freebsd",
  target_os = "android"
))]
const MAP_STACK: MapFlags = MapFlags::STACK;

#[cfg(not(any(
  all(target_os = "linux", not(target_arch = "mips")),
  target_os = "freebsd",
  target_os = "android"
)))]
const MAP_STACK: MapFlags = MapFlags::empty();

#[cfg(any(target_os = "linux", target_os = "android"))]
const MAP_POPULATE: MapFlags = MapFlags::POPULATE;

#[cfg(not(any(target_os = "linux", target_os = "android")))]
const MAP_POPULATE: MapFlags = MapFlags::empty();

#[cfg(any(target_os = "linux", target_os = "android"))]
const MAP_HUGETLB: MapFlags = MapFlags::HUGETLB;

#[cfg(not(any(target_os = "linux", target_os = "android")))]
const MAP_HUGETLB: MapFlags = MapFlags::empty();

// Linux encodes an explicit hugepage size in bits 26..=31 of the mmap flags
// (log2 of the page size). rustix v1 does not expose MAP_HUGE_MASK /
// MAP_HUGE_SHIFT publicly, so source them from linux-raw-sys directly.
#[cfg(target_os = "linux")]
use linux_raw_sys::general::{MAP_HUGE_MASK, MAP_HUGE_SHIFT};

#[cfg(any(
  target_os = "linux",
  target_os = "android",
  target_vendor = "apple",
  target_os = "netbsd",
  target_os = "solaris",
  target_os = "illumos",
))]
const MAP_NORESERVE: MapFlags = MapFlags::NORESERVE;

#[cfg(not(any(
  target_os = "linux",
  target_os = "android",
  target_vendor = "apple",
  target_os = "netbsd",
  target_os = "solaris",
  target_os = "illumos",
)))]
const MAP_NORESERVE: MapFlags = MapFlags::empty();

pub struct MmapInner {
  ptr: *mut core::ffi::c_void,
  len: usize,
}

impl MmapInner {
  /// Creates a new `MmapInner`.
  ///
  /// This is a thin wrapper around the `mmap` system call.
  fn new(
    len: usize,
    prot: ProtFlags,
    flags: MapFlags,
    file: RawFd,
    offset: u64,
  ) -> io::Result<MmapInner> {
    let alignment = offset % page_size() as u64;
    let aligned_offset = offset - alignment;

    let (map_len, map_offset) = Self::adjust_mmap_params(len, alignment as usize)?;

    unsafe {
      let fd = BorrowedFd::borrow_raw(file);
      match rustix::mm::mmap(ptr::null_mut(), map_len, prot, flags, fd, aligned_offset) {
        Ok(ptr) => Ok(Self::from_raw_parts(ptr, len, map_offset)),
        Err(e) => Err(io::Error::from_raw_os_error(e.raw_os_error())),
      }
    }
  }

  fn adjust_mmap_params(len: usize, alignment: usize) -> io::Result<(usize, usize)> {
    // Rust's slice cannot be larger than isize::MAX.
    // See https://doc.rust-lang.org/std/slice/fn.from_raw_parts.html
    //
    // This is not a problem on 64-bit targets, but on 32-bit one
    // having a file or an anonymous mapping larger than 2GB is quite normal
    // and we have to prevent it.
    //
    // The code below is essentially the same as in Rust's std:
    // https://github.com/rust-lang/rust/blob/db78ab70a88a0a5e89031d7ee4eccec835dcdbde/library/alloc/src/raw_vec.rs#L495
    if std::mem::size_of::<usize>() < 8 && len > isize::MAX as usize {
      return Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "memory map length overflows isize",
      ));
    }

    let map_len = len + alignment;
    let map_offset = alignment;

    // `mmap` does not support zero-size mappings. POSIX defines:
    //
    // https://pubs.opengroup.org/onlinepubs/9699919799/functions/mmap.html
    // > If `len` is zero, `mmap()` shall fail and no mapping shall be established.
    //
    // So if we would create such a mapping, crate a one-byte mapping instead:
    let map_len = map_len.max(1);

    // Note that in that case `MmapInner::len` is still set to zero,
    // and `Mmap` will still dereferences to an empty slice.
    //
    // If this mapping is backed by an empty file, we create a mapping larger than the file.
    // This is unusual but well-defined. On the same man page, POSIX further defines:
    //
    // > The `mmap()` function can be used to map a region of memory that is larger
    // > than the current size of the object.
    //
    // (The object here is the file.)
    //
    // > Memory access within the mapping but beyond the current end of the underlying
    // > objects may result in SIGBUS signals being sent to the process. The reason for this
    // > is that the size of the object can be manipulated by other processes and can change
    // > at any moment. The implementation should tell the application that a memory reference
    // > is outside the object where this can be detected; otherwise, written data may be lost
    // > and read data may not reflect actual data in the object.
    //
    // Because `MmapInner::len` is not incremented, this increment of `aligned_len`
    // will not allow accesses past the end of the file and will not cause SIGBUS.
    //
    // (SIGBUS is still possible by mapping a non-empty file and then truncating it
    // to a shorter size, but that is unrelated to this handling of empty files.)
    Ok((map_len, map_offset))
  }

  /// Get the current memory mapping as a `(ptr, map_len, offset)` tuple.
  ///
  /// Note that `map_len` is the length of the memory mapping itself and
  /// _not_ the one that would be passed to `from_raw_parts`.
  fn as_mmap_params(&self) -> (*mut core::ffi::c_void, usize, usize) {
    let offset = self.ptr as usize % page_size();
    let len = self.len + offset;

    // There are two possible memory layouts we could have, depending on
    // the length and offset passed when constructing this instance:
    //
    // 1. The "normal" memory layout looks like this:
    //
    //         |<------------------>|<---------------------->|
    //     mmap ptr    offset      ptr     public slice
    //
    //    That is, we have
    //    - The start of the page-aligned memory mapping returned by mmap,
    //      followed by,
    //    - Some number of bytes that are memory mapped but ignored since
    //      they are before the byte offset requested by the user, followed
    //      by,
    //    - The actual memory mapped slice requested by the user.
    //
    //    This maps cleanly to a (ptr, len, offset) tuple.
    //
    // 2. Then, we have the case where the user requested a zero-length
    //    memory mapping. mmap(2) does not support zero-length mappings so
    //    this crate works around that by actually making a mapping of
    //    length one. This means that we have
    //    - A length zero slice, followed by,
    //    - A single memory mapped byte
    //
    //    Note that this only happens if the offset within the page is also
    //    zero. Otherwise, we have a memory map of offset bytes and not a
    //    zero-length memory map.
    //
    //    This doesn't fit cleanly into a (ptr, len, offset) tuple. Instead,
    //    we fudge it slightly: a zero-length memory map turns into a
    //    mapping of length one and can't be told apart outside of this
    //    method without knowing the original length.
    if len == 0 {
      (self.ptr, 1, 0)
    } else {
      (unsafe { self.ptr.offset(-(offset as isize)) }, len, offset)
    }
  }

  /// Construct this `MmapInner` from its raw components
  ///
  /// # Safety
  ///
  /// - `ptr` must point to the start of memory mapping that can be freed
  ///   using `munmap(2)` (i.e. returned by `mmap(2)` or `mremap(2)`)
  /// - The memory mapping at `ptr` must have a length of `len + offset`.
  /// - If `len + offset == 0` then the memory mapping must be of length 1.
  /// - `offset` must be less than the current page size.
  unsafe fn from_raw_parts(ptr: *mut core::ffi::c_void, len: usize, offset: usize) -> Self {
    debug_assert_eq!(ptr as usize % page_size(), 0, "ptr not page-aligned");
    debug_assert!(offset < page_size(), "offset larger than page size");

    Self {
      ptr: ptr.add(offset),
      len,
    }
  }

  pub fn map(
    len: usize,
    file: RawFd,
    offset: u64,
    populate: bool,
    no_reserve: bool,
  ) -> io::Result<MmapInner> {
    let populate = if populate {
      MAP_POPULATE
    } else {
      MapFlags::empty()
    };
    let no_reserve = if no_reserve {
      MAP_NORESERVE
    } else {
      MapFlags::empty()
    };
    MmapInner::new(
      len,
      ProtFlags::READ,
      MapFlags::SHARED | populate | no_reserve,
      file,
      offset,
    )
  }

  pub fn map_exec(
    len: usize,
    file: RawFd,
    offset: u64,
    populate: bool,
    no_reserve: bool,
  ) -> io::Result<MmapInner> {
    let populate = if populate {
      MAP_POPULATE
    } else {
      MapFlags::empty()
    };
    let no_reserve = if no_reserve {
      MAP_NORESERVE
    } else {
      MapFlags::empty()
    };
    MmapInner::new(
      len,
      ProtFlags::READ | ProtFlags::EXEC,
      MapFlags::SHARED | populate | no_reserve,
      file,
      offset,
    )
  }

  pub fn map_mut(
    len: usize,
    file: RawFd,
    offset: u64,
    populate: bool,
    no_reserve: bool,
  ) -> io::Result<MmapInner> {
    let populate = if populate {
      MAP_POPULATE
    } else {
      MapFlags::empty()
    };
    let no_reserve = if no_reserve {
      MAP_NORESERVE
    } else {
      MapFlags::empty()
    };
    MmapInner::new(
      len,
      ProtFlags::READ | ProtFlags::WRITE,
      MapFlags::SHARED | populate | no_reserve,
      file,
      offset,
    )
  }

  pub fn map_copy(
    len: usize,
    file: RawFd,
    offset: u64,
    populate: bool,
    no_reserve: bool,
  ) -> io::Result<MmapInner> {
    let populate = if populate {
      MAP_POPULATE
    } else {
      MapFlags::empty()
    };
    let no_reserve = if no_reserve {
      MAP_NORESERVE
    } else {
      MapFlags::empty()
    };
    MmapInner::new(
      len,
      ProtFlags::READ | ProtFlags::WRITE,
      MapFlags::PRIVATE | populate | no_reserve,
      file,
      offset,
    )
  }

  pub fn map_copy_read_only(
    len: usize,
    file: RawFd,
    offset: u64,
    populate: bool,
    no_reserve: bool,
  ) -> io::Result<MmapInner> {
    let populate = if populate {
      MAP_POPULATE
    } else {
      MapFlags::empty()
    };
    let no_reserve = if no_reserve {
      MAP_NORESERVE
    } else {
      MapFlags::empty()
    };
    MmapInner::new(
      len,
      ProtFlags::READ,
      MapFlags::PRIVATE | populate | no_reserve,
      file,
      offset,
    )
  }

  /// Open an anonymous memory map.
  pub fn map_anon(
    len: usize,
    stack: bool,
    populate: bool,
    huge: Option<u8>,
    no_reserve: bool,
  ) -> io::Result<MmapInner> {
    let stack = if stack { MAP_STACK } else { MapFlags::empty() };
    let populate = if populate {
      MAP_POPULATE
    } else {
      MapFlags::empty()
    };
    let hugetlb = if huge.is_some() {
      MAP_HUGETLB
    } else {
      MapFlags::empty()
    };
    let hugetlb_size = huge_size_flag(huge);
    let no_reserve = if no_reserve {
      MAP_NORESERVE
    } else {
      MapFlags::empty()
    };

    // `mmap_anonymous` passes `MAP_ANON` implicitly; callers only supply
    // auxiliary flags. Adjust for the same zero-length workaround as the
    // file-backed path.
    let (map_len, _) = Self::adjust_mmap_params(len, 0)?;

    unsafe {
      match rustix::mm::mmap_anonymous(
        ptr::null_mut(),
        map_len,
        ProtFlags::READ | ProtFlags::WRITE,
        MapFlags::PRIVATE | stack | populate | hugetlb | hugetlb_size | no_reserve,
      ) {
        Ok(ptr) => Ok(Self::from_raw_parts(ptr, len, 0)),
        Err(e) => Err(io::Error::from_raw_os_error(e.raw_os_error())),
      }
    }
  }

  pub fn flush(&self, offset: usize, len: usize) -> io::Result<()> {
    let alignment = (self.ptr as usize + offset) % page_size();
    let offset = offset as isize - alignment as isize;
    let len = len + alignment;
    unsafe { rustix::mm::msync(self.ptr.offset(offset), len, MsyncFlags::SYNC) }
      .map_err(|e| io::Error::from_raw_os_error(e.raw_os_error()))
  }

  pub fn flush_async(&self, offset: usize, len: usize) -> io::Result<()> {
    let alignment = (self.ptr as usize + offset) % page_size();
    let offset = offset as isize - alignment as isize;
    let len = len + alignment;
    unsafe { rustix::mm::msync(self.ptr.offset(offset), len, MsyncFlags::ASYNC) }
      .map_err(|e| io::Error::from_raw_os_error(e.raw_os_error()))
  }

  fn mprotect(&mut self, prot: MprotectFlags) -> io::Result<()> {
    unsafe {
      let alignment = self.ptr as usize % page_size();
      let ptr = self.ptr.offset(-(alignment as isize));
      let len = self.len + alignment;
      let len = len.max(1);
      rustix::mm::mprotect(ptr, len, prot)
        .map_err(|e| io::Error::from_raw_os_error(e.raw_os_error()))
    }
  }

  pub fn make_read_only(&mut self) -> io::Result<()> {
    self.mprotect(MprotectFlags::READ)
  }

  pub fn make_exec(&mut self) -> io::Result<()> {
    self.mprotect(MprotectFlags::READ | MprotectFlags::EXEC)
  }

  pub fn make_mut(&mut self) -> io::Result<()> {
    self.mprotect(MprotectFlags::READ | MprotectFlags::WRITE)
  }

  #[inline]
  pub fn ptr(&self) -> *const u8 {
    self.ptr as *const u8
  }

  #[inline]
  pub fn mut_ptr(&mut self) -> *mut u8 {
    self.ptr.cast()
  }

  #[inline]
  pub fn len(&self) -> usize {
    self.len
  }

  pub(crate) fn advise(
    &self,
    advice: crate::advice::RawAdvice,
    offset: usize,
    len: usize,
  ) -> io::Result<()> {
    use crate::advice::RawAdvice;

    let alignment = (self.ptr as usize + offset) % page_size();
    let offset = offset as isize - alignment as isize;
    let len = len + alignment;
    unsafe {
      let ptr = self.ptr.offset(offset);

      match advice {
        // Apple goes through libc because rustix v1 does not model the
        // Darwin-specific `MADV_*` advice values this crate exposes.
        #[cfg(target_vendor = "apple")]
        RawAdvice::Libc(value) => {
          if libc::madvise(ptr, len, value) == 0 {
            Ok(())
          } else {
            Err(io::Error::last_os_error())
          }
        }
        #[cfg(not(target_vendor = "apple"))]
        RawAdvice::Rustix(value) => {
          // The AIX signature of `madvise()` differs from the POSIX
          // specification, which expects `void *` as the type of
          // the `addr` argument, whereas AIX uses `caddr_t` (i.e.
          // `char *`). rustix normalizes this on its side, but the
          // pointer cast below is a no-op on non-AIX platforms.
          #[cfg(target_os = "aix")]
          let ptr = ptr as *mut u8 as *mut core::ffi::c_void;

          rustix::mm::madvise(ptr, len, value)
            .map_err(|e| io::Error::from_raw_os_error(e.raw_os_error()))
        }
      }
    }
  }

  #[cfg(target_os = "linux")]
  pub fn remap(&mut self, new_len: usize, options: crate::RemapOptions) -> io::Result<()> {
    let (old_ptr, old_len, offset) = self.as_mmap_params();
    let (map_len, offset) = Self::adjust_mmap_params(new_len, offset)?;

    unsafe {
      match rustix::mm::mremap(old_ptr, old_len, map_len, options.into_flags()) {
        Ok(new_ptr) => {
          // We explicitly don't drop self since the pointer within is no longer valid.
          ptr::write(self, Self::from_raw_parts(new_ptr, new_len, offset));
          Ok(())
        }
        Err(e) => Err(io::Error::from_raw_os_error(e.raw_os_error())),
      }
    }
  }

  pub fn lock(&self) -> io::Result<()> {
    unsafe {
      rustix::mm::mlock(self.ptr, self.len)
        .map_err(|e| io::Error::from_raw_os_error(e.raw_os_error()))
    }
  }

  pub fn unlock(&self) -> io::Result<()> {
    unsafe {
      rustix::mm::munlock(self.ptr, self.len)
        .map_err(|e| io::Error::from_raw_os_error(e.raw_os_error()))
    }
  }
}

impl Drop for MmapInner {
  fn drop(&mut self) {
    let (ptr, len, _) = self.as_mmap_params();

    // Any errors during unmapping/closing are ignored as the only way
    // to report them would be through panicking which is highly discouraged
    // in Drop impls, c.f. https://github.com/rust-lang/lang-team/issues/97
    unsafe {
      let _ = rustix::mm::munmap(ptr, len);
    }
  }
}

unsafe impl Sync for MmapInner {}
unsafe impl Send for MmapInner {}

#[cfg(target_os = "linux")]
fn huge_size_flag(huge: Option<u8>) -> MapFlags {
  match huge {
    Some(mask) => {
      let bits = (u32::from(mask) & MAP_HUGE_MASK) << MAP_HUGE_SHIFT;
      MapFlags::from_bits_retain(bits)
    }
    None => MapFlags::empty(),
  }
}

#[cfg(not(target_os = "linux"))]
fn huge_size_flag(_huge: Option<u8>) -> MapFlags {
  MapFlags::empty()
}

fn page_size() -> usize {
  static PAGE_SIZE: AtomicUsize = AtomicUsize::new(0);

  match PAGE_SIZE.load(Ordering::Relaxed) {
    0 => {
      let page_size = rustix::param::page_size();

      PAGE_SIZE.store(page_size, Ordering::Relaxed);

      page_size
    }
    page_size => page_size,
  }
}

pub fn file_len(file: RawFd) -> io::Result<u64> {
  // SAFETY: We must not close the passed-in fd by dropping the File we create,
  // we ensure this by immediately wrapping it in a ManuallyDrop.
  unsafe {
    let file = ManuallyDrop::new(File::from_raw_fd(file));
    Ok(file.metadata()?.len())
  }
}
