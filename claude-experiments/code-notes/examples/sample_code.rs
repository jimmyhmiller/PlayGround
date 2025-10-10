// Sample Rust code for demonstrating code-notes
// This file shows various scenarios where notes could be useful

use std::collections::HashMap;

/// Main authentication function
/// This is a complex function that handles user authentication
pub fn authenticate_user(username: &str, password: &str) -> Result<String, AuthError> {
    // NOTE: You could add a code-note here explaining the authentication flow
    let user_db = load_user_database()?;

    if let Some(user) = user_db.get(username) {
        if verify_password(password, &user.password_hash) {
            let token = generate_session_token(username);
            Ok(token)
        } else {
            Err(AuthError::InvalidPassword)
        }
    } else {
        Err(AuthError::UserNotFound)
    }
}

fn load_user_database() -> Result<HashMap<String, User>, AuthError> {
    // NOTE: This could have a note about the database schema
    let mut db = HashMap::new();
    db.insert(
        "alice".to_string(),
        User {
            username: "alice".to_string(),
            password_hash: hash_password("secret123"),
            email: "alice@example.com".to_string(),
        },
    );
    Ok(db)
}

fn verify_password(password: &str, hash: &str) -> bool {
    // NOTE: Could note that this uses bcrypt internally
    hash_password(password) == hash
}

fn hash_password(password: &str) -> String {
    // Simplified for example - in reality would use proper hashing
    format!("hashed_{}", password)
}

fn generate_session_token(username: &str) -> String {
    // NOTE: Could explain token generation strategy
    format!("token_for_{}", username)
}

#[derive(Debug)]
struct User {
    username: String,
    password_hash: String,
    email: String,
}

#[derive(Debug)]
pub enum AuthError {
    UserNotFound,
    InvalidPassword,
    DatabaseError,
}

impl std::error::Error for AuthError {}

impl std::fmt::Display for AuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            AuthError::UserNotFound => write!(f, "User not found"),
            AuthError::InvalidPassword => write!(f, "Invalid password"),
            AuthError::DatabaseError => write!(f, "Database error"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_authenticate_valid_user() {
        // NOTE: Could add notes about test scenarios
        let result = authenticate_user("alice", "secret123");
        assert!(result.is_ok());
    }

    #[test]
    fn test_authenticate_invalid_password() {
        let result = authenticate_user("alice", "wrong_password");
        assert!(matches!(result, Err(AuthError::InvalidPassword)));
    }

    #[test]
    fn test_authenticate_nonexistent_user() {
        let result = authenticate_user("bob", "any_password");
        assert!(matches!(result, Err(AuthError::UserNotFound)));
    }
}
