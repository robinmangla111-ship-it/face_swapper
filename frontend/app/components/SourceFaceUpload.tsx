"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  addLibraryCharacter,
  deleteLibraryCharacter,
  getLibraryCharacters,
  getLibraryImageUrl,
  selectLibraryCharacter,
  uploadSourceFace,
  type Character,
  type LibraryCharacter,
  type UploadSourceFaceResponse,
} from "../lib/api";

interface SourceFaceUploadProps {
  targetCharacter: Character;
  onUploadComplete: (data: UploadSourceFaceResponse) => void;
  onError: (msg: string) => void;
}

const CATEGORIES = ["All", "Avengers", "Hollywood", "Bollywood", "Mythology", "Custom"];

export default function SourceFaceUpload({
  targetCharacter,
  onUploadComplete,
  onError,
}: SourceFaceUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [sourceFace, setSourceFace] = useState<UploadSourceFaceResponse | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Character Library state
  const [libraryChars, setLibraryChars] = useState<LibraryCharacter[]>([]);
  const [loadingLib, setLoadingLib] = useState(true);
  const [activeCategory, setActiveCategory] = useState("All");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectingId, setSelectingId] = useState<string | null>(null);

  // Add New Character form state
  const [showAddForm, setShowAddForm] = useState(false);
  const [newName, setNewName] = useState("");
  const [newCategory, setNewCategory] = useState("Avengers");
  const [newSwapMode, setNewSwapMode] = useState("auto");
  const [newFile, setNewFile] = useState<File | null>(null);
  const [savingNew, setSavingNew] = useState(false);
  const newCharFileRef = useRef<HTMLInputElement>(null);

  const loadLibrary = useCallback(async () => {
    try {
      setLoadingLib(true);
      const list = await getLibraryCharacters();
      setLibraryChars(list);
    } catch {
      // ignore initial library fetch errors
    } finally {
      setLoadingLib(false);
    }
  }, []);

  useEffect(() => {
    loadLibrary();
  }, [loadLibrary]);

  const handleUpload = useCallback(
    async (file: File) => {
      const ext = file.name.split(".").pop()?.toLowerCase();
      if (!["jpg", "jpeg", "png", "webp"].includes(ext || "")) {
        onError("Please upload a JPG, PNG, or WEBP image");
        return;
      }
      setUploading(true);
      setProgress(0);

      try {
        const data = await uploadSourceFace(file, (pct) => setProgress(pct));
        setSourceFace(data);
        setUploading(false);
        onUploadComplete(data);
      } catch (err: unknown) {
        setUploading(false);
        const message = err instanceof Error ? err.message : "Upload failed";
        onError(message);
      }
    },
    [onUploadComplete, onError]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleUpload(file);
    },
    [handleUpload]
  );

  const handleSelectFromLibrary = async (char: LibraryCharacter) => {
    try {
      setSelectingId(char.id);
      const data = await selectLibraryCharacter(char.id);
      setSourceFace(data);
      onUploadComplete(data);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Failed to select character";
      onError(message);
    } finally {
      setSelectingId(null);
    }
  };

  const handleAddCharacter = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newFile || !newName.trim()) {
      onError("Please provide both a character name and an image");
      return;
    }
    try {
      setSavingNew(true);
      const created = await addLibraryCharacter(newFile, newName.trim(), newCategory, newSwapMode);
      setLibraryChars((prev) => [...prev, created]);
      setNewName("");
      setNewFile(null);
      setShowAddForm(false);
      // Automatically select the newly added character
      await handleSelectFromLibrary(created);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Failed to add character";
      onError(message);
    } finally {
      setSavingNew(false);
    }
  };

  const handleDeleteCharacter = async (e: React.MouseEvent, charId: string) => {
    e.stopPropagation();
    try {
      await deleteLibraryCharacter(charId);
      setLibraryChars((prev) => prev.filter((c) => c.id !== charId));
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Failed to delete character";
      onError(message);
    }
  };

  const filteredChars = libraryChars.filter((c) => {
    const matchesCat =
      activeCategory === "All" || c.category.toLowerCase() === activeCategory.toLowerCase();
    const matchesSearch =
      !searchQuery.trim() || c.name.toLowerCase().includes(searchQuery.trim().toLowerCase());
    return matchesCat && matchesSearch;
  });

  return (
    <div className="animate-fade-in space-y-8">
      {/* Side-by-side comparison */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Target face */}
        <div className="card p-5 space-y-3">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-red-400" />
            <h3 className="text-sm font-semibold text-zinc-300 uppercase tracking-wider">
              Target — Person to Replace
            </h3>
          </div>
          <div className="aspect-square max-w-[200px] mx-auto rounded-xl overflow-hidden bg-zinc-800">
            <img
              src={`data:image/jpeg;base64,${targetCharacter.thumbnail}`}
              alt={targetCharacter.label}
              className="w-full h-full object-cover"
            />
          </div>
          <p className="text-center text-sm text-zinc-400">{targetCharacter.label}</p>
        </div>

        {/* Source face */}
        <div className="card p-5 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-green-400" />
              <h3 className="text-sm font-semibold text-zinc-300 uppercase tracking-wider">
                Source — New Face
              </h3>
            </div>
            {sourceFace && (
              <button
                onClick={() => setSourceFace(null)}
                className="text-xs text-zinc-400 hover:text-zinc-200 underline"
              >
                Change
              </button>
            )}
          </div>

          {sourceFace ? (
            <>
              <div className="aspect-square max-w-[200px] mx-auto rounded-xl overflow-hidden bg-zinc-800 relative">
                <img
                  src={`data:image/jpeg;base64,${sourceFace.thumbnail}`}
                  alt="Source face"
                  className="w-full h-full object-cover"
                />
                <div className="absolute bottom-2 right-2 px-2 py-0.5 bg-black/75 backdrop-blur-sm rounded-lg text-[11px] font-bold text-green-400">
                  {sourceFace.swap_mode === "mask_overlay"
                    ? "🎭 Mask / Helmet"
                    : `${(sourceFace.confidence * 100).toFixed(0)}% Face`}
                </div>
              </div>
              <p className="text-center text-sm text-zinc-300 font-medium">{sourceFace.filename}</p>
              {/* Mode selector */}
              <div className="flex flex-wrap justify-center gap-2 pt-1">
                {(["neural", "mask_overlay"] as const).map((mode) => (
                  <button
                    key={mode}
                    type="button"
                    onClick={() => {
                      const updated = { ...sourceFace, swap_mode: mode };
                      setSourceFace(updated);
                      onUploadComplete(updated);
                    }}
                    className={`px-3 py-1 rounded-lg text-xs font-medium border transition-all ${
                      (sourceFace.swap_mode || "neural") === mode
                        ? "bg-brand-500/20 border-brand-500 text-brand-300"
                        : "bg-zinc-800/60 border-zinc-700 text-zinc-400 hover:text-zinc-200"
                    }`}
                  >
                    {mode === "neural"
                      ? "👤 Neural Face Swap"
                      : "🎭 Head / Helmet Overlay"}
                  </button>
                ))}
              </div>
            </>
          ) : (
            <div
              className={`drop-zone text-center aspect-square max-w-[200px] mx-auto flex flex-col items-center justify-center ${
                isDragging ? "active" : ""
              }`}
              onDrop={handleDrop}
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".jpg,.jpeg,.png,.webp"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) handleUpload(file);
                }}
              />

              {uploading ? (
                <div className="space-y-3">
                  <div className="w-10 h-10 mx-auto rounded-full border-3 border-zinc-700 border-t-brand-500 animate-spin" />
                  <p className="text-xs text-zinc-400">{progress}%</p>
                </div>
              ) : (
                <div className="space-y-2 px-3">
                  <svg
                    className="w-8 h-8 mx-auto text-zinc-500"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={1.5}
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
                  </svg>
                  <p className="text-xs text-zinc-400">Upload custom face or pick below</p>
                  <p className="text-[11px] text-zinc-600">JPG, PNG, WEBP</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Character Library */}
      <div className="card p-5 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h3 className="text-base font-bold text-zinc-100">
              🎬 Character &amp; Celebrity Library
            </h3>
            <p className="text-xs text-zinc-400">
              Pick from Avengers, Hollywood, Bollywood, Mythology — or add your own!
            </p>
          </div>
          <button
            type="button"
            onClick={() => setShowAddForm((v) => !v)}
            className="px-3.5 py-2 rounded-xl bg-brand-600 hover:bg-brand-500 text-white text-xs font-semibold transition-all shadow-lg shadow-brand-500/20 self-start sm:self-auto"
          >
            {showAddForm ? "✕ Close" : "+ Add New Character"}
          </button>
        </div>

        {/* Add New Character Inline Form */}
        {showAddForm && (
          <form
            onSubmit={handleAddCharacter}
            className="p-4 rounded-xl bg-zinc-900/90 border border-zinc-700/70 space-y-4 animate-fade-in"
          >
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div>
                <label className="block text-xs text-zinc-400 mb-1">Character Name</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. Spider-Man, Ranbir Kapoor"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg bg-zinc-800 border border-zinc-700 text-sm text-zinc-100 focus:outline-none focus:border-brand-500"
                />
              </div>
              <div>
                <label className="block text-xs text-zinc-400 mb-1">Category</label>
                <select
                  value={newCategory}
                  onChange={(e) => setNewCategory(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg bg-zinc-800 border border-zinc-700 text-sm text-zinc-100 focus:outline-none focus:border-brand-500"
                >
                  {CATEGORIES.filter((c) => c !== "All").map((cat) => (
                    <option key={cat} value={cat}>
                      {cat}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-xs text-zinc-400 mb-1">Swap Style</label>
                <select
                  value={newSwapMode}
                  onChange={(e) => setNewSwapMode(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg bg-zinc-800 border border-zinc-700 text-sm text-zinc-100 focus:outline-none focus:border-brand-500"
                >
                  <option value="auto">Auto Detect (Recommended)</option>
                  <option value="neural">Human / Celebrity Face</option>
                  <option value="mask_overlay">Head Mask / Helmet / Stylized</option>
                </select>
              </div>
            </div>

            <div className="flex flex-col sm:flex-row items-center justify-between gap-3">
              <div className="flex items-center gap-3 w-full sm:w-auto">
                <input
                  ref={newCharFileRef}
                  type="file"
                  accept=".jpg,.jpeg,.png,.webp"
                  className="hidden"
                  onChange={(e) => setNewFile(e.target.files?.[0] || null)}
                />
                <button
                  type="button"
                  onClick={() => newCharFileRef.current?.click()}
                  className="px-3 py-2 rounded-lg bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 text-xs text-zinc-200"
                >
                  {newFile ? `📷 ${newFile.name}` : "Choose Image File..."}
                </button>
              </div>
              <button
                type="submit"
                disabled={savingNew}
                className="px-4 py-2 rounded-lg bg-green-600 hover:bg-green-500 disabled:opacity-50 text-white text-xs font-semibold"
              >
                {savingNew ? "Saving..." : "Save to Library & Select"}
              </button>
            </div>
          </form>
        )}

        {/* Filter Bar */}
        <div className="flex flex-col sm:flex-row gap-3 items-stretch sm:items-center justify-between">
          <div className="flex flex-wrap gap-1.5">
            {CATEGORIES.map((cat) => (
              <button
                key={cat}
                type="button"
                onClick={() => setActiveCategory(cat)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  activeCategory === cat
                    ? "bg-brand-600 text-white shadow-sm"
                    : "bg-zinc-800/80 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800"
                }`}
              >
                {cat}
              </button>
            ))}
          </div>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search character..."
            className="px-3 py-1.5 rounded-lg bg-zinc-900 border border-zinc-700/80 text-xs text-zinc-200 placeholder-zinc-500 focus:outline-none focus:border-brand-500 w-full sm:w-52"
          />
        </div>

        {/* Character Grid */}
        {loadingLib ? (
          <div className="py-8 text-center text-xs text-zinc-500">Loading character library...</div>
        ) : filteredChars.length === 0 ? (
          <div className="py-8 text-center text-xs text-zinc-500">
            No characters match this filter. Click &quot;+ Add New Character&quot; to add one!
          </div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3 max-h-[360px] overflow-y-auto pr-1">
            {filteredChars.map((char) => {
              const isSelected = sourceFace?.face_id === `lib_${char.id}`;
              const isBusy = selectingId === char.id;
              return (
                <div
                  key={char.id}
                  onClick={() => !isBusy && handleSelectFromLibrary(char)}
                  className={`group relative rounded-xl overflow-hidden border cursor-pointer transition-all bg-zinc-900/70 ${
                    isSelected
                      ? "border-green-500 ring-2 ring-green-500/30"
                      : "border-zinc-800 hover:border-brand-500/60"
                  }`}
                >
                  <div className="aspect-square bg-zinc-800 overflow-hidden relative">
                    <img
                      src={getLibraryImageUrl(char.image_url)}
                      alt={char.name}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                    />
                    {isBusy && (
                      <div className="absolute inset-0 bg-black/60 flex items-center justify-center">
                        <div className="w-6 h-6 rounded-full border-2 border-zinc-400 border-t-white animate-spin" />
                      </div>
                    )}
                    <span className="absolute top-1.5 left-1.5 px-1.5 py-0.5 rounded bg-black/70 text-[10px] text-zinc-300">
                      {char.swap_mode === "mask_overlay" ? "🎭 Mask" : char.category}
                    </span>
                    {char.is_custom && (
                      <button
                        type="button"
                        onClick={(e) => handleDeleteCharacter(e, char.id)}
                        title="Remove custom character"
                        className="absolute top-1.5 right-1.5 w-5 h-5 rounded-full bg-red-900/80 hover:bg-red-600 text-white text-[10px] opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center"
                      >
                        ✕
                      </button>
                    )}
                  </div>
                  <div className="p-2">
                    <p className="text-xs font-medium text-zinc-200 truncate">{char.name}</p>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
