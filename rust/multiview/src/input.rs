//! The input contract: what the photos and intrinsics must satisfy before any work starts.
//!
//! Checked up front so that a bad input set fails in milliseconds instead of after the
//! first correspondence run.

use std::sync::Arc;

use pixelmap::{Photo, MIN_DIMENSION};

use crate::calib::Intrinsics;
use crate::error::Error;
use crate::types::ViewId;

/// The fewest photos a reconstruction accepts.
///
/// Two views are what `pixelmap_model_3d` is for. A third is what makes a
/// cycle-consistency check possible, and a registration rigid enough to be trusted.
pub const MIN_VIEWS: usize = 3;

/// Checks that there are enough photos, that they all have the same dimensions and are
/// large enough for pixelmap, and that the intrinsics are usable.
///
/// Returns the photos' common `(width, height)`.
///
/// # Errors
/// [`Error::TooFewPhotos`], [`Error::PhotoTooSmall`], [`Error::SizeMismatch`] or
/// [`Error::InvalidIntrinsics`], checked in that order and reporting the first photo that
/// fails.
pub fn validate(photos: &[Arc<Photo>], intrinsics: &Intrinsics) -> Result<(usize, usize), Error> {
    if photos.len() < MIN_VIEWS {
        return Err(Error::TooFewPhotos {
            found: photos.len(),
            minimum: MIN_VIEWS,
        });
    }

    let expected = (photos[0].width(), photos[0].height());
    for (index, photo) in photos.iter().enumerate() {
        let view = ViewId(index as u32);
        let dimensions = (photo.width(), photo.height());
        if dimensions.0 < MIN_DIMENSION || dimensions.1 < MIN_DIMENSION {
            return Err(Error::PhotoTooSmall {
                view,
                dimensions,
                minimum: MIN_DIMENSION,
            });
        }
        if dimensions != expected {
            return Err(Error::SizeMismatch {
                view,
                expected,
                found: dimensions,
            });
        }
    }

    if !intrinsics.is_valid() {
        return Err(Error::InvalidIntrinsics);
    }
    Ok(expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blank(width: usize, height: usize) -> Arc<Photo> {
        Arc::new(Photo::from_rgba(width, height, vec![0; width * height * 4]).unwrap())
    }

    #[test]
    fn accepts_matching_photos() {
        let photos = vec![blank(64, 48); 4];
        let k = Intrinsics::estimated(64, 48);
        assert_eq!(validate(&photos, &k).unwrap(), (64, 48));
    }

    #[test]
    fn rejects_too_few_photos() {
        let photos = vec![blank(64, 48); 2];
        let k = Intrinsics::estimated(64, 48);
        assert!(matches!(
            validate(&photos, &k),
            Err(Error::TooFewPhotos { found: 2, .. })
        ));
    }

    #[test]
    fn names_the_photo_that_differs() {
        let photos = vec![blank(64, 48), blank(64, 48), blank(48, 64)];
        let k = Intrinsics::estimated(64, 48);
        let error = validate(&photos, &k).unwrap_err();
        assert!(matches!(
            error,
            Error::SizeMismatch {
                view: ViewId(2),
                found: (48, 64),
                ..
            }
        ));
        assert_eq!(error.stage(), crate::Stage::Input);
    }

    #[test]
    fn rejects_small_photos_and_bad_intrinsics() {
        let photos = vec![blank(64, 16); 3];
        let k = Intrinsics::estimated(64, 16);
        assert!(matches!(
            validate(&photos, &k),
            Err(Error::PhotoTooSmall {
                view: ViewId(0),
                ..
            })
        ));

        let photos = vec![blank(64, 48); 3];
        let mut k = Intrinsics::estimated(64, 48);
        k.fy = -1.0;
        assert!(matches!(
            validate(&photos, &k),
            Err(Error::InvalidIntrinsics)
        ));
    }
}
