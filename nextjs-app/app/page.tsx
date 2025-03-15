"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Loader2, Upload } from "lucide-react"
import Image from "next/image"
import { Alert, AlertDescription } from "@/components/ui/alert"

export default function VideoUploadForm() {
  const [file, setFile] = useState<File | null>(null)
  const [sentiment, setSentiment] = useState<string>("")
  const [isLoading, setIsLoading] = useState(false)
  const [responseImage, setResponseImage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0]

      // Check file type
      const fileType = selectedFile.type
      if (fileType !== "video/mp4" && fileType !== "video/webm") {
        setError("Only MP4 and WEBM files are allowed")
        return
      }

      // Check file size (2GB = 2 * 1024 * 1024 * 1024 bytes)
      if (selectedFile.size > 200 * 1024 * 1024) {
        setError("File size must be less than 2GB")
        return
      }

      setFile(selectedFile)
      setError(null)
    }
  }

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const selectedFile = e.dataTransfer.files[0]

      // Check file type
      const fileType = selectedFile.type
      if (fileType !== "video/mp4" && fileType !== "video/webm") {
        setError("Only MP4 and WEBM files are allowed")
        return
      }

      // Check file size (2GB = 2 * 1024 * 1024 * 1024 bytes)
      if (selectedFile.size > 2 * 1024 * 1024 * 1024) {
        setError("File size must be less than 2GB")
        return
      }

      setFile(selectedFile)
      setError(null)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
  e.preventDefault();

  if (!file) {
    setError("Please select a video file");
    return;
  }

  if (!sentiment) {
    setError("Please select a sentiment");
    return;
  }

  setIsLoading(true);
  setError(null);
  setResponseImage(null);

  const formData = new FormData();
  formData.append("video", file);
  formData.append("sentiment", sentiment);

  try {
    // Submit the video for processing.
    const response = await fetch("/image-api/process_video", {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      throw new Error("Request failed");
    }

    // Retrieve the job id from the API response.
    const { job_id: jobId } = await response.json();

    // Poll the job status every 10 seconds.
    const pollJobStatus = async () => {
      try {
        const statusRes = await fetch(`/image-api/job_status/${jobId}`);
        if (!statusRes.ok) {
          throw new Error("Job status request failed");
        }
        const jobData = await statusRes.json();
        if (jobData.status === "inprogress") {
          // If still in progress, poll again in 10 seconds.
          setTimeout(pollJobStatus, 10000);
        } else if (jobData.status === "failed") {
          setError(jobData.error || "Job failed");
          setIsLoading(false);
        } else if (jobData.status === "completed") {
          // When completed, create an image URL from the base64 string.
          const imageUrl = `data:image/png;base64,${jobData.image}`;
          setResponseImage(imageUrl);
          setIsLoading(false);
        } else {
          // In case of unexpected status, poll again.
          setTimeout(pollJobStatus, 10000);
        }
      } catch (pollError) {
        setError("Something went wrong while checking job status");
        setIsLoading(false);
        console.error(pollError);
      }
    };

    pollJobStatus();
  } catch (err) {
    setError("Something went wrong");
    console.error(err);
    setIsLoading(false);
  }
  };


  return (
    <div className="container mx-auto py-10 px-4">
      <Card className="max-w-2xl mx-auto">
        <CardHeader>
          <CardTitle className="text-2xl">Sentiment Based Thumbnail</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Video Upload Area */}
            <div
              className="border-2 border-dashed rounded-lg p-6 text-center cursor-pointer hover:bg-muted/50 transition-colors"
              onClick={() => fileInputRef.current?.click()}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
            >
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept="video/mp4,video/webm"
                className="hidden"
              />
              <div className="flex flex-col items-center justify-center gap-2">
                <Upload className="h-10 w-10 text-muted-foreground" />
                <p className="text-sm font-medium">
                  {file ? file.name : "Click to upload or drag and drop a video file"}
                </p>
                <p className="text-xs text-muted-foreground">
                  {file ? `${(file.size / (1024 * 1024)).toFixed(2)} MB` : "MP4, WEBM up to 200MB"}
                </p>
              </div>
            </div>

            {/* Sentiment Dropdown */}
            <div className="space-y-2">
              <label htmlFor="sentiment" className="text-sm font-medium">
                Select Sentiment
              </label>
              <Select value={sentiment} onValueChange={setSentiment}>
                <SelectTrigger id="sentiment">
                  <SelectValue placeholder="Select sentiment" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="positive">Positive</SelectItem>
                  <SelectItem value="negative">Negative</SelectItem>
                  <SelectItem value="neutral">Neutral</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Submit Button */}
            <Button type="submit" className="w-full" disabled={isLoading || !file || !sentiment}>
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Processing...
                </>
              ) : (
                "Submit"
              )}
            </Button>

            {/* Error Message */}
            {error && (
              <Alert variant="destructive">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {/* Response Image */}
            {responseImage && (
              <div className="mt-6">
                <h3 className="text-lg font-medium mb-2">Result:</h3>
                <div className="border rounded-lg overflow-hidden">
                  <Image
                    src={responseImage || "/placeholder.svg"}
                    alt="Analysis Result"
                    width={600}
                    height={400}
                    className="w-full h-auto"
                  />
                </div>
              </div>
            )}
          </form>
        </CardContent>
      </Card>
    </div>
  )
}

