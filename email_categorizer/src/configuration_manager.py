class ConfigurationManager:
    _instance = None  # Private class variable to hold the single instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigurationManager, cls).__new__(cls)
            cls._instance._settings = {}
        return cls._instance

    def set_config(self, key, value):
        """Sets a configuration setting."""
        self._settings[key] = value
        print(f"Configuration set: {key} = {value}")

    def get_config(self, key, default=None):
        """Retrieves a configuration setting, returning a default value if not found."""
        return self._settings.get(key, default)

    def remove_config(self, key):
        """Removes a configuration setting if it exists."""
        if key in self._settings:
            del self._settings[key]
            print(f"Configuration removed: {key}")

    def clear_all(self):
        """Clears all configuration settings."""
        self._settings.clear()
        print("All configurations cleared.")


# Example usage
if __name__ == "__main__":
    # Get the ConfigurationManager instance
    config_manager1 = ConfigurationManager()
    
    # Set some configuration values
    config_manager1.set_config("email_classification_threshold", 0.8)
    config_manager1.set_config("default_classifier", "NaiveBayes")

    # Retrieve a configuration value
    print("Threshold:", config_manager1.get_config("email_classification_threshold"))
    print("Default Classifier:", config_manager1.get_config("default_classifier"))

    # Verify Singleton behavior
    config_manager2 = ConfigurationManager()
    print("Config manager 1 and 2 are the same:", config_manager1 is config_manager2)

    # Both instances should share the same settings
    config_manager2.set_config("log_level", "DEBUG")
    print("Log level from manager 1:", config_manager1.get_config("log_level"))
